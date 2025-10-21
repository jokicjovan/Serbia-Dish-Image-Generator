# train.py
import os, math, argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from dataset import CaptionImageSet
from diffaug import diff_augment
from models import Generator, Discriminator
from ema import EMA
from tqdm import tqdm

def hinge_d(real_logits, fake_logits, mis_logits=None, mis_weight=0.5):
    loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    if mis_logits is not None:
        loss = loss + mis_weight * F.relu(1 + mis_logits).mean()
    return loss

def hinge_g(fake_logits):
    return -fake_logits.mean()

def r1_penalty(real_x, real_logits):
    # \gamma/2 * ||grad D(x)||^2 ; we will multiply by gamma outside
    grad = torch.autograd.grad(
        outputs=real_logits.sum(), inputs=real_x,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    penalty = grad.pow(2).reshape(grad.size(0), -1).sum(dim=1).mean()
    return penalty

def save_samples(path, imgs):
    grid = make_grid((imgs.clamp(-1,1)+1)/2, nrow=int(math.sqrt(imgs.size(0))+0.5))
    save_image(grid, path)

def main(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----- Data
    ds = CaptionImageSet(args.data_root, size=args.img_size)
    emb_dim = ds[0][1].numel()
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, pin_memory=True)

    # ----- Models
    G = Generator(z_dim=args.z_dim, cond_in=emb_dim, cond_hidden=args.cond_dim,
                  base_ch=args.base_ch, out_size=args.img_size).to(device)
    D = Discriminator(cond_in=emb_dim, base_ch=args.base_ch, in_size=args.img_size).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.0, 0.9))

    ema = EMA(G, decay=args.ema)

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    os.makedirs(args.out_dir, exist_ok=True)
    fixed = next(iter(dl))
    fixed_e = fixed[1][:args.n_sample].to(device)
    fixed_z = torch.randn(args.n_sample, args.z_dim, device=device)

    step = 0
    pbar = tqdm(total=args.iters, desc="training")

    while step < args.iters:
        for x, e in dl:
            x, e = x.to(device, non_blocking=True), e.to(device, non_blocking=True)
            B = x.size(0)

            # Create mismatched text by rolling (no fixed points)
            e_mis = torch.roll(e, shifts=1, dims=0)

            # ----------------- D update -----------------
            for _ in range(args.n_disc):
                z = torch.randn(B, args.z_dim, device=device)
                with torch.no_grad():
                    x_fake = G(z, e).detach()

                xr, xf = x, x_fake
                if args.diffaugment:
                    xr = diff_augment(xr)
                    xf = diff_augment(xf)

                xr.requires_grad_(True)
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    real_logits = D(xr, e)
                    fake_logits = D(xf, e)
                    mis_logits  = D(xr, e_mis) if args.use_mismatch else None
                    d_loss = hinge_d(real_logits, fake_logits, mis_logits, mis_weight=args.mismatch_w)

                optD.zero_grad(set_to_none=True)
                if device=="cuda":
                    scaler.scale(d_loss).backward()
                else:
                    d_loss.backward()

                # Lazy R1
                if (step % args.r1_every) == 0:
                    with torch.cuda.amp.autocast(enabled=False):
                        r1 = r1_penalty(xr, real_logits)
                    (args.r1_gamma/2) * r1.backward()

                if device=="cuda":
                    scaler.step(optD)
                    scaler.update()
                else:
                    optD.step()

            # ----------------- G update -----------------
            z = torch.randn(B, args.z_dim, device=device)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                x_fake = G(z, e)
                xf = diff_augment(x_fake) if args.diffaugment else x_fake
                g_loss = hinge_g(D(xf, e))

            optG.zero_grad(set_to_none=True)
            if device=="cuda":
                scaler.scale(g_loss).backward()
                scaler.step(optG); scaler.update()
            else:
                g_loss.backward(); optG.step()

            # EMA update
            ema.update(G)

            # ----------------- logging / samples -----------------
            if (step % args.sample_every) == 0:
                # sample with EMA weights
                ema.apply_shadow(G)
                with torch.no_grad():
                    z = fixed_z
                    efix = fixed_e
                    imgs = G(z, efix)
                    save_samples(os.path.join(args.out_dir, f"samples_{step:07d}.png"), imgs)
                ema.restore(G)

            if (step % args.ckpt_every) == 0 and step > 0:
                torch.save({
                    "G": G.state_dict(), "D": D.state_dict(),
                    "optG": optG.state_dict(), "optD": optD.state_dict(),
                    "ema": ema.shadow, "step": step,
                    "args": vars(args)
                }, os.path.join(args.out_dir, f"ckpt_{step:07d}.pt"))

            step += 1
            pbar.update(1)
            pbar.set_postfix({"d_loss": float(d_loss.detach().cpu()), "g_loss": float(g_loss.detach().cpu())})

            if step >= args.iters:
                break

    pbar.close()
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed")
    ap.add_argument("--out_dir", type=str, default="runs/cgan")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--cond_dim", type=int, default=256)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--n_disc", type=int, default=1)
    ap.add_argument("--g_lr", type=float, default=2e-4)
    ap.add_argument("--d_lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--diffaugment", action="store_true")
    ap.add_argument("--use_mismatch", action="store_true")
    ap.add_argument("--mismatch_w", type=float, default=0.5)
    ap.add_argument("--r1_every", type=int, default=16)
    ap.add_argument("--r1_gamma", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--sample_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--n_sample", type=int, default=8)
    args = ap.parse_args()
    main(args)
