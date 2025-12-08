import os, math, argparse, random
import numpy as np
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- your local modules ----
from dataset import CaptionImageSet
from diffaug import diff_augment
from models import Generator, Discriminator
from ema import EMA

# ---- optional torchvision utils (with safe fallbacks) ----
try:
    from torchvision.utils import make_grid, save_image  # type: ignore
except Exception:
    def make_grid(t, nrow=8, normalize=False, value_range=None):
        N, C, H, W = t.size()
        nrow = min(nrow, N)
        ncol = math.ceil(N / nrow)
        grid = t.new_zeros(C, ncol * H, nrow * W)
        k = 0
        if normalize and value_range is not None:
            lo, hi = value_range
            t = (t - lo) / max(1e-6, (hi - lo))
            t = t.clamp(0, 1)
        for i in range(ncol):
            for j in range(nrow):
                if k >= N: break
                grid[:, i*H:(i+1)*H, j*W:(j+1)*W] = t[k]
                k += 1
        return grid

    def save_image(img, fp):
        import imageio.v2 as imageio
        arr = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype('uint8')
        imageio.imwrite(fp, arr)

# ---------------------- losses & regularizers ----------------------
def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, mis_logits: torch.Tensor | None = None, mis_w: float = 0.5):
    loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    if mis_logits is not None:
        loss = loss + mis_w * F.relu(1.0 + mis_logits).mean()
    return loss

def hinge_g_loss(fake_logits: torch.Tensor):
    return (-fake_logits).mean()

def r1_penalty(x_real: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
    grad_real, = torch.autograd.grad(outputs=real_logits.sum(), inputs=x_real, create_graph=True)
    return grad_real.pow(2).view(x_real.size(0), -1).sum(1).mean()

# ---------------------- adaptive controller ----------------------
@dataclass
class AdaptiveCfg:
    enabled: bool = True
    target_hinge: float = 1.6     # desired D hinge loss mid-range
    hi_margin: float = 0.35       # D too weak if > target + margin
    lo_margin: float = 0.9        # D too strong if < target - margin
    d_lr_min: float = 1e-5
    d_lr_max: float = 3e-4
    d_lr_step: float = 2.0        # multiplicative change
    r1_min: float = 0.05
    r1_max: float = 5.0
    r1_step: float = 2.0
    n_disc_min: int = 1
    n_disc_max: int = 3
    window: int = 200             # EWMA window
    cool_down: int = 400          # steps between adjustments

class EWMA:
    def _init_(self, beta: float): self.beta, self.val = beta, None
    def update(self, x: float): self.val = x if self.val is None else self.beta*self.val + (1-self.beta)*x; return self.val

class AdaptiveController:
    def _init_(self, cfg: AdaptiveCfg):
        self.cfg = cfg
        self.ewma_d = EWMA(beta=1 - 1/cfg.window)
        self.last_tune = -10**9

    def maybe_tune(self, step: int, optD: torch.optim.Optimizer, r1_gamma: float, n_disc: int) -> Tuple[float, int]:
        if not self.cfg.enabled or step - self.last_tune < self.cfg.cool_down or self.ewma_d.val is None:
            return r1_gamma, n_disc
        cur = self.ewma_d.val
        cur_lr = optD.param_groups[0]['lr']
        tuned = False

        # D too weak -> boost D
        if cur > (self.cfg.target_hinge + self.cfg.hi_margin):
            new_lr = min(cur_lr * self.cfg.d_lr_step, self.cfg.d_lr_max)
            if new_lr > cur_lr * 1.01:
                for pg in optD.param_groups: pg['lr'] = new_lr
                tuned = True
            if n_disc < self.cfg.n_disc_max:
                n_disc += 1; tuned = True
            if r1_gamma > self.cfg.r1_min:
                r1_gamma = max(r1_gamma / self.cfg.r1_step, self.cfg.r1_min); tuned = True

        # D too strong -> nerf D
        elif cur < (self.cfg.target_hinge - self.cfg.lo_margin):
            new_lr = max(cur_lr / self.cfg.d_lr_step, self.cfg.d_lr_min)
            if new_lr < cur_lr / 1.01:
                for pg in optD.param_groups: pg['lr'] = new_lr
                tuned = True
            if n_disc > self.cfg.n_disc_min:
                n_disc -= 1; tuned = True
            if r1_gamma < self.cfg.r1_max:
                r1_gamma = min(r1_gamma * self.cfg.r1_step, self.cfg.r1_max); tuned = True

        if tuned: self.last_tune = step
        return r1_gamma, n_disc

# ---------------------- dataset helper ----------------------
class DatasetWrapper(CaptionImageSet):
    """
    Thin wrapper around your CaptionImageSet to provide:
      - sample_fixed_conditions(n): loads n embeddings from disk (L2-normalized)
    """
    def sample_fixed_conditions(self, n: int):
        import random
        ids = list(self.ids) if hasattr(self, 'ids') else list(range(len(self)))
        random.shuffle(ids)
        ids = ids[:n]
        embs = []
        for _id in ids:
            emb_path = os.path.join(self.emb_dir, f"{_id}.npy")
            e = np.load(emb_path).astype("float32")
            e /= (np.linalg.norm(e) + 1e-8)          # L2 norm
            embs.append(torch.from_numpy(e))
        return torch.stack(embs, dim=0)

# ---------------------- training ----------------------
def set_seed(s): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--image_size', type=int, default=64)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--iters', type=int, default=20000)
    # optim
    ap.add_argument('--g_lr', type=float, default=2e-4)
    ap.add_argument('--d_lr', type=float, default=2e-4)
    ap.add_argument('--betas', type=float, nargs=2, default=(0.0, 0.9))
    # reg
    ap.add_argument('--r1_gamma', type=float, default=0.5)
    ap.add_argument('--r1_every', type=int, default=16)
    # ratio
    ap.add_argument('--n_disc', type=int, default=2)
    # ema
    ap.add_argument('--ema_decay', type=float, default=0.999)
    # diffaug
    ap.add_argument('--diffaugment', action='store_true')
    ap.add_argument('--diffaug_policy', type=str, default='color,translation,cutout')
    ap.add_argument('--diffaug_warmup', type=int, default=2000)
    # adaptive
    ap.add_argument('--adaptive', action='store_true')
    # misc
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fp16', action='store_true')
    # save cadence
    ap.add_argument('--sample_every', type=int, default=500)
    ap.add_argument('--ckpt_every', type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    ds = DatasetWrapper(root=args.data_root, size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    it = iter(loader)

    # models
    G = Generator().to(device)
    D = Discriminator().to(device)
    ema = EMA(G, decay=args.ema_decay)

    optG = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=tuple(args.betas))
    optD = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=tuple(args.betas))
    scalerG = torch.cuda.amp.GradScaler(enabled=args.fp16)
    scalerD = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # fixed conds/noise for sampling
    z_dim = getattr(G, 'z_dim', 128)
    fixed_z = torch.randn(64, z_dim, device=device)
    fixed_c = ds.sample_fixed_conditions(64).to(device)

    controller = AdaptiveController(AdaptiveCfg())
    controller.cfg.enabled = args.adaptive

    step = 0
    r1_gamma = args.r1_gamma
    n_disc = args.n_disc

    while step < args.iters:
        # ------------------- D updates -------------------
        for _ in range(n_disc):
            try:
                xr, c = next(it)
            except StopIteration:
                it = iter(loader)
                xr, c = next(it)

            xr = xr.to(device, non_blocking=True)
            c = F.normalize(c.to(device, non_blocking=True), dim=1)   # L2 normalize conds
            b = xr.size(0)
            z = torch.randn(b, z_dim, device=device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                xf = G(z, c).detach()

                # DiffAug warmup
                if args.diffaugment and step >= args.diffaug_warmup:
                    xr = diff_augment(xr, policy=args.diffaug_policy)
                    xf = diff_augment(xf, policy=args.diffaug_policy)

                real_logits = D(xr, c)
                fake_logits = D(xf, c)

                mis_logits = None
                if hasattr(args, 'use_mismatch') and args.use_mismatch:
                    c_mis = c[torch.randperm(b)]
                    mis_logits = D(xr, c_mis)

                d_loss = hinge_d_loss(real_logits, fake_logits, mis_logits, getattr(args, 'mismatch_w', 0.5))

            optD.zero_grad(set_to_none=True)
            if device == 'cuda':
                scalerD.scale(d_loss).backward()
            else:
                d_loss.backward()

            # Lazy R1 in fp32
            if (step % args.r1_every) == 0:
                xr.requires_grad_(True)
                with torch.cuda.amp.autocast(enabled=False):
                    r1 = r1_penalty(xr, D(xr, c))
                (r1_gamma * 0.5 * r1).backward()
                xr.requires_grad_(False)

            if device == 'cuda':
                scalerD.step(optD); scalerD.update()
            else:
                optD.step()

            controller.ewma_d.update(d_loss.detach().item())

        # ------------------- G update -------------------
        try:
            _, c = next(it)
        except StopIteration:
            it = iter(loader)
            _, c = next(it)
        c = F.normalize(c.to(device, non_blocking=True), dim=1)
        b = c.size(0)
        z = torch.randn(b, z_dim, device=device)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            xf = G(z, c)
            if args.diffaugment and step >= args.diffaug_warmup:
                xf = diff_augment(xf, policy=args.diffaug_policy)
            g_loss = hinge_g_loss(D(xf, c))

        optG.zero_grad(set_to_none=True)
        if device == 'cuda':
            scalerG.scale(g_loss).backward()
            scalerG.step(optG); scalerG.update()
        else:
            g_loss.backward(); optG.step()

        ema.update(G)

        # ---- adaptive retune ----
        if args.adaptive and step > 500:
            r1_gamma, n_disc = controller.maybe_tune(step, optD, r1_gamma, n_disc)

        # ---- logs / ckpts / samples ----
        if (step % 50) == 0:
            with torch.no_grad():
                d_real = D(xr.to(device), c.to(device)).mean().item()
                d_fake = D(xf.to(device), c.to(device)).mean().item()
            print(f"step {step:05d} | D {d_loss.item():.3f} | G {g_loss.item():.3f} | D(real) {d_real:.3f} | D(fake) {d_fake:.3f} | r1 {r1_gamma:.3f} | n_disc {n_disc} | d_lr {optD.param_groups[0]['lr']:.2e}")

        if step > 0 and (step % args.ckpt_every) == 0:
            ck = {'G': G.state_dict(), 'D': D.state_dict(),
                  'opt_g': optG.state_dict(), 'opt_d': optD.state_dict(), 'step': step}
            torch.save(ck, os.path.join(args.out_dir, f'ckpt_{step:06d}.pt'))
            # EMA checkpoint for inference
            ema.copy_to(G)
            torch.save({'G_ema': G.state_dict(), 'step': step}, os.path.join(args.out_dir, f'ckpt_ema_{step:06d}.pt'))

        if (step % args.sample_every) == 0:
            with torch.no_grad():
                ema.copy_to(G)
                z = fixed_z
                c_fix = fixed_c.to(device)
                fake = G(z, c_fix).clamp(-1, 1)
                grid = make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
                save_image(grid, os.path.join(args.out_dir, f'sample_{step:06d}.png'))

        step += 1

    print("Training complete.")

if __name__ == "__main__":
    main()