import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import device, cuda, save, load, no_grad, cat, clamp, randn
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import CaptionImageSet
from model import ConvCVAE

np.random.seed(2)
torch.manual_seed(2)

device = device("cuda" if cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_dataset(args):
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    train_dataset = CaptionImageSet(
        root=args.data_root,
        img_size=args.img_size,
        test_split=args.test_split,
        is_test=False
    )

    test_dataset = CaptionImageSet(
        root=args.data_root,
        img_size=args.img_size,
        test_split=args.test_split,
        is_test=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


def generate_img_grid(images, save_path, title):
    images = clamp((images * 0.5 + 0.5), 0, 1)
    grid = make_grid(images, nrow=4)
    np_grid = grid.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(np_grid, (1, 2, 0)))
    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def loss_function(recon_x, x, mean, log_var, beta=1.0, warmup_factor=1.0):
    recon_loss = mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    kl_loss = clamp(kl_loss, max=1e6)
    total_loss = recon_loss + (beta * warmup_factor * kl_loss)

    return total_loss, recon_loss, kl_loss


def get_warmup_factor(epoch, warmup_epochs=10):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0


def train_epoch(model, train_loader, optimizer, beta, epoch, warmup_epochs):
    model.train()
    tr_loss, tr_recon_loss, tr_kl = 0, 0, 0

    warmup_factor = get_warmup_factor(epoch, warmup_epochs)

    for batch_idx, (data, clip_emb) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        clip_emb = clip_emb.to(device)

        optimizer.zero_grad()
        recon_batch, mean, log_var = model(data, clip_emb)
        loss, recon_loss, kl_loss = loss_function(recon_batch, data, mean, log_var, beta, warmup_factor)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tr_loss += loss.item()
        tr_recon_loss += recon_loss.item()
        tr_kl += kl_loss.item()

    n_samples = len(train_loader.dataset)
    return tr_loss / n_samples, tr_recon_loss / n_samples, tr_kl / n_samples, warmup_factor


def validate(model, val_loader, beta):
    model.eval()
    val_loss, val_recon, val_kl = 0, 0, 0

    with no_grad():
        for data, clip_emb in val_loader:
            data = data.to(device)
            clip_emb = clip_emb.to(device)
            recon_batch, mean, log_var = model(data, clip_emb)
            loss, recon_loss, kl_loss = loss_function(recon_batch, data, mean, log_var, beta, warmup_factor=1.0)
            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()

    n_samples = len(val_loader.dataset)
    return val_loss / n_samples, val_recon / n_samples, val_kl / n_samples


def train(args):

    def save_plots_images(label):
        data_iter = iter(test_loader)
        images, embeddings = next(data_iter)

        model.eval()
        with no_grad():
            images = images.to(device)
            embeddings = embeddings.to(device)
            recon, _, _ = model(images, embeddings)

        comparison = cat([images[:8].cpu(), recon[:8].cpu()])
        save_path = os.path.join(result_dir, f"reconstruction_epoch_{label}.png")
        generate_img_grid(comparison, save_path, f"Reconstruction - Epoch {label}")

        generated = model.generate(embeddings[:16], num_samples=16)
        save_path = os.path.join(result_dir, f"generated_epoch_{label}.png")
        generate_img_grid(generated, save_path, f"Generated - Epoch {label}")

        plot_losses(history, os.path.join(loss_dir, f"losses_epoch_{label}.png"))

    start_time_total = time.perf_counter()
    dt = datetime.now()
    dt_str = dt.strftime("%Y-%m-%d_%H-%M-%S")

    output_root = f'{dt_str}_{args.output_dir}'
    os.makedirs(output_root, exist_ok=True)

    checkpoint_dir = os.path.join(output_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    result_dir = os.path.join(output_root, "results")
    os.makedirs(result_dir, exist_ok=True)

    loss_dir = os.path.join(output_root, "losses")
    os.makedirs(loss_dir, exist_ok=True)

    train_loader, test_loader = load_dataset(args)

    model = ConvCVAE(img_dim=[args.img_size, args.img_size, 3],latent_dim=args.latent_dim).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"\nModel Architecture:")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  CLIP dim: 512")
    print(f"  Image size: {args.img_size}")
    print(f"  Beta: {args.beta}")
    print(f"  KL warmup epochs: {args.warmup_epochs}")

    epoch_idx = 0
    min_loss = float('inf')

    if args.checkpoint:
        checkpoint = load(args.checkpoint, map_location=device)
        epoch_idx = checkpoint['epoch_idx'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        min_loss = checkpoint.get('min_loss', float('inf'))
        print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_loss': [],
        'val_recon': [],
        'val_kl': []
    }

    for epoch in range(epoch_idx, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_recon, train_kl, warmup = train_epoch(
            model, train_loader, optimizer, args.beta, epoch, args.warmup_epochs
        )

        val_loss, val_recon, val_kl = validate(model, test_loader, args.beta)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)
        history['val_loss'].append(val_loss)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)

        print(f"Warmup: {warmup:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}")

        # Save best model
        if val_loss < min_loss:
            min_loss = val_loss
            save({
                'epoch_idx': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'min_loss': min_loss,
            }, os.path.join(checkpoint_dir, "best_checkpoint.pt"))
            print ('Best checkpoint saved')
            save_plots_images('best')


        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
            save({
                'epoch_idx': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'min_loss': min_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            save_plots_images(str(epoch + 1))

    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    save({
        'epoch_idx': args.epochs - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'min_loss': min_loss,
    }, final_path)

    plot_losses(history, os.path.join(loss_dir, "losses_final.png"))

    total_time = time.perf_counter() - start_time_total
    print(f"\n" + "=" * 70)
    print(f"TRAINING COMPLETE")
    print(f"Best validation loss: {min_loss:.4f}")
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    print(f"Results saved to: {output_root}")
    print("=" * 70)




def plot_losses(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_recon'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_recon'], 'r-', label='Val')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history['train_kl'], 'b-', label='Train')
    axes[2].plot(epochs, history['val_kl'], 'r-', label='Val')
    axes[2].set_title('KL Divergence')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP-CVAE on food images")
    parser.add_argument("--data_root", type=str, default="data/processed",
                        help="Root directory containing images and embeds folders")
    parser.add_argument("--output_dir", type=str, default="results_pytorch",
                        help="Output directory for checkpoints and samples")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Image size (height and width)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Fraction of data to use for testing")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Latent dimension size")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Beta parameter for KL loss weighting")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of epochs for KL warmup")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training")

    args = parser.parse_args()
    train(args)
