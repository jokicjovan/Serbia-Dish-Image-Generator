# train.py
import os, math, argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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
    grad = torch.autograd.grad(
        outputs=real_logits.sum(), inputs=real_x,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    penalty = grad.pow(2).reshape(grad.size(0), -1).sum(dim=1).mean()
    return penalty

def save_samples(path, imgs):
    grid = make_grid((imgs.clamp(-1,1)+1)/2, nrow=int(math.sqrt(imgs.size(0))+0.5))
    save_image(grid, path)

def plot_losses(log_data, output_dir):
    """Plot and save training curves."""
    steps = [x['step'] for x in log_data]
    d_losses = [x['d_loss'] for x in log_data]
    g_losses = [x['g_loss'] for x in log_data]
    real_logits = [x.get('real_logits_mean', 0) for x in log_data]
    fake_logits = [x.get('fake_logits_mean', 0) for x in log_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(steps, d_losses, label='D Loss', alpha=0.7)
    axes[0, 0].plot(steps, g_losses, label='G Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # D loss only (zoomed)
    axes[0, 1].plot(steps, d_losses, color='blue', alpha=0.7)
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('D Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # G loss only (zoomed)
    axes[1, 0].plot(steps, g_losses, color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('G Loss')
    axes[1, 0].set_title('Generator Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Logits (real vs fake scores)
    axes[1, 1].plot(steps, real_logits, label='Real Logits', alpha=0.7)
    axes[1, 1].plot(steps, fake_logits, label='Fake Logits', alpha=0.7)
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Logits')
    axes[1, 1].set_title('Discriminator Outputs')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

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

    scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        G.load_state_dict(checkpoint["G"])
        D.load_state_dict(checkpoint["D"])
        optG.load_state_dict(checkpoint["optG"])
        optD.load_state_dict(checkpoint["optD"])
        ema.shadow = checkpoint["ema"]
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")

    os.makedirs(args.out_dir, exist_ok=True)
    fixed = next(iter(dl))
    fixed_e = fixed[1][:args.n_sample].to(device)
    fixed_z = torch.randn(args.n_sample, args.z_dim, device=device)

    # Logging
    log_data = []
    log_file = os.path.join(args.out_dir, 'training_log.json')

    # Load existing log data if resuming
    if args.resume_from and os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            print(f"Loaded {len(log_data)} existing log entries")
        except:
            print("Could not load existing log data, starting fresh")
            log_data = []

    step = start_step
    pbar = tqdm(total=args.iters, desc="training", initial=start_step)

    while step < args.iters:
        for x, e in dl:
            x, e = x.to(device, non_blocking=True), e.to(device, non_blocking=True)
            B = x.size(0)

            # Create mismatched text by random permutation (better entropy than fixed roll)
            perm_idx = torch.randperm(B)
            # Ensure at least some mismatches by rerolling if permutation is identity
            while torch.equal(perm_idx, torch.arange(B)):
                perm_idx = torch.randperm(B)
            e_mis = e[perm_idx]

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
                with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                    real_logits = D(xr, e)
                    fake_logits = D(xf, e)
                    mis_logits  = D(xr, e_mis) if args.use_mismatch else None
                    d_loss = hinge_d(real_logits, fake_logits, mis_logits, mis_weight=args.mismatch_w)

                # Compute total discriminator loss (including R1 if needed)
                total_d_loss = d_loss
                if (step % args.r1_every) == 0:
                    with torch.amp.autocast('cuda', enabled=False):
                        r1 = r1_penalty(xr, real_logits)
                    total_d_loss = d_loss + (args.r1_gamma/2) * r1

                optD.zero_grad(set_to_none=True)
                if device=="cuda":
                    scaler.scale(total_d_loss).backward()
                else:
                    total_d_loss.backward()

                if device=="cuda":
                    if args.grad_clip > 0:
                        scaler.unscale_(optD)
                        torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
                    scaler.step(optD)
                    scaler.update()
                else:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
                    optD.step()

            # ----------------- G update -----------------
            z = torch.randn(B, args.z_dim, device=device)
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                x_fake = G(z, e)
                xf = diff_augment(x_fake) if args.diffaugment else x_fake
                g_loss = hinge_g(D(xf, e))

            optG.zero_grad(set_to_none=True)
            if device=="cuda":
                scaler.scale(g_loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optG)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
                scaler.step(optG); scaler.update()
            else:
                g_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
                optG.step()

            # EMA update
            ema.update(G)

            # ----------------- Logging -----------------
            if step % args.log_every == 0:
                log_entry = {
                    'step': step,
                    'd_loss': float(d_loss.detach().cpu()),
                    'g_loss': float(g_loss.detach().cpu()),
                    'real_logits_mean': float(real_logits.mean().detach().cpu()),
                    'fake_logits_mean': float(fake_logits.mean().detach().cpu()),
                }
                log_data.append(log_entry)
                
                # Save log file
                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)
                
                # Plot curves
                if step > 0 and step % args.plot_every == 0:
                    plot_losses(log_data, args.out_dir)

            # ----------------- Samples -----------------
            if (step % args.sample_every) == 0:
                ema.apply_shadow(G)
                with torch.no_grad():
                    imgs = G(fixed_z, fixed_e)
                    save_samples(os.path.join(args.out_dir, f"samples_{step:07d}.png"), imgs)
                ema.restore(G)

            # ----------------- Checkpoints -----------------
            if (step % args.ckpt_every) == 0 and step > 0:
                torch.save({
                    "G": G.state_dict(), "D": D.state_dict(),
                    "optG": optG.state_dict(), "optD": optD.state_dict(),
                    "ema": ema.shadow, "step": step,
                    "args": vars(args)
                }, os.path.join(args.out_dir, f"ckpt_{step:07d}.pt"))

            step += 1
            pbar.update(1)
            pbar.set_postfix({
                "d_loss": f"{float(d_loss.detach().cpu()):.4f}", 
                "g_loss": f"{float(g_loss.detach().cpu()):.4f}"
            })

            if step >= args.iters:
                break

    # Final plot
    plot_losses(log_data, args.out_dir)
    pbar.close()
    print(f"\nTraining completed! Check {args.out_dir} for results.")

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
    ap.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping norm (0 = disabled)")
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
    ap.add_argument("--log_every", type=int, default=10, help="Log metrics every N steps")
    ap.add_argument("--plot_every", type=int, default=500, help="Update plots every N steps")
    ap.add_argument("--resume_from", type=str, default="", help="Path to checkpoint file to resume from")
    args = ap.parse_args()
    main(args)