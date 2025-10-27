import argparse
import os

from dataset import CaptionImageSet
from keras.src.optimizers import Adam
import tensorflow as tf
import time
import numpy as np
from image_generation import image_generation, image_reconstruction
from utils import train_step, plot_losses, save_data
from datetime import datetime
from model import ConvCVAE, Decoder, Encoder

tf.random.set_seed(2)
np.random.seed(2)


def plot_test_reconstruction(epoch, save_folder, model, sample):
    filename = f"reconstruction_epoch_{epoch}.png"
    full_path = os.path.join(save_folder, filename)
    test_images, test_labels = sample
    image_reconstruction(model, test_images, test_labels, save_path=full_path)


def generate_sample_images(prompts, save_folder, model, num_images=32):
    for prompt in prompts:
        print(f"\nGenerating images for: '{prompt}'")
        image_generation(model, prompt, num_images=num_images, save_path=save_folder)


def load_dataset(args):
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    train_dataset = CaptionImageSet(
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch,
        shuffle=True,
        test_split=args.test_split,
        is_test=False
    )

    test_dataset = CaptionImageSet(
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch,
        shuffle=False,
        test_split=args.test_split,
        is_test=True
    )
    return train_dataset, test_dataset


def train(args):
    start_time_total = time.perf_counter()
    dt = datetime.now()
    dt_str = dt.strftime("%Y-%m-%d_%H-%M-%S")

    if args.pretrained_model:
        checkpoint_name = args.pretrained_model
    else:
        checkpoint_name = dt_str

    checkpoint_root = os.path.join(".", "checkpoints", checkpoint_name)
    checkpoint_prefix = "model"
    save_prefix = os.path.join(checkpoint_root, checkpoint_prefix)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    result_folder = os.path.join(".", args.out_dir, f"result_{checkpoint_name}")
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    encoder = Encoder(args.latent_dim, concat_input_and_condition=True)
    decoder = Decoder(batch_size=args.batch)
    model = ConvCVAE(
        encoder,
        decoder,
        label_dim=args.label_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        batch_size=args.batch,
        image_dim=[args.img_size, args.img_size, 3]
    )

    optimizer = Adam(learning_rate=args.lr)

    print(f"Model created:")
    print(f"  - Latent dim: {args.latent_dim}")
    print(f"  - Label dim: {args.label_dim}")
    print(f"  - Beta: {args.beta}")
    print(f"  - Learning rate: {args.lr}")

    checkpoint = tf.train.Checkpoint(module=model, optimizer=optimizer)

    if args.checkpoint_path:
        checkpoint.restore(args.checkpoint_path).expect_partial()
        print(f"Loaded checkpoint: {args.checkpoint_path}")
    else:
        latest = tf.train.latest_checkpoint(checkpoint_root)
        if latest is not None:
            checkpoint.restore(latest).expect_partial()
            print(f"Restored latest checkpoint: {latest}")
        else:
            print("No checkpoint found, starting from scratch")

    print(f"Results will be saved to: {result_folder}")
    print(f"Checkpoints will be saved to: {checkpoint_root}")
    train_dataset, test_dataset = load_dataset(args)

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    n_batches = len(train_dataset)
    print(f"Number of epochs: {args.iters}")
    print(f"Number of batches per epoch: {n_batches}")

    # Loss tracking
    epoch_losses = {
        'total': [],
        'reconstruction': [],
        'latent': []
    }

    # Training loop
    for epoch in range(args.iters):
        epoch_start_time = time.perf_counter()

        batch_losses = []
        batch_recon_losses = []
        batch_latent_losses = []

        # Shuffle dataset at start of epoch
        train_dataset.on_epoch_end()

        # Batch loop
        for batch_idx in range(n_batches):
            images, labels = train_dataset[batch_idx]

            total_loss, recon_loss, latent_loss = train_step(
                (images, labels),
                model,
                optimizer
            )

            batch_losses.append(total_loss)
            batch_recon_losses.append(recon_loss)
            batch_latent_losses.append(latent_loss)

            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{n_batches} - "
                      f"Loss: {total_loss:.4f}, "
                      f"Recon: {recon_loss:.4f}, "
                      f"Latent: {latent_loss:.4f}")

        # Calculate epoch averages
        avg_loss = np.mean(batch_losses)
        avg_recon = np.mean(batch_recon_losses)
        avg_latent = np.mean(batch_latent_losses)

        epoch_losses['total'].append(avg_loss)
        epoch_losses['reconstruction'].append(avg_recon)
        epoch_losses['latent'].append(avg_latent)

        epoch_time = time.perf_counter() - epoch_start_time

        print(f"\nEpoch {epoch + 1}/{args.iters} - Time: {epoch_time:.2f}s")
        print(f"  Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Latent: {avg_latent:.4f}")

        print(f"Epoch {epoch + 1}: "
              f"loss={avg_loss:.4f}, "
              f"recon={avg_recon:.4f}, "
              f"latent={avg_latent:.4f}, "
              f"time={epoch_time:.2f}s\n")

        # Save progress
        if (epoch + 1) % args.n_sample == 0:
            checkpoint.save(save_prefix)
            print(f"  Checkpoint saved: {save_prefix}")

            plot_test_reconstruction(epoch + 1, result_folder, model, test_dataset[0])

            # Plot loss curves
            loss_plot_path = os.path.join(result_folder, f"losses_epoch_{epoch + 1}.png")
            plot_losses(epoch_losses, save_path=loss_plot_path)

    # Save final model
    if args.iters % args.n_sample != 0:
        checkpoint.save(save_prefix)
        print(f"\nFinal checkpoint saved: {save_prefix}")

    # Final reconstruction and generation
    plot_test_reconstruction(args.iters, result_folder, model, test_dataset[0])

    # Save loss data
    save_data(os.path.join(result_folder, "losses"), epoch_losses)

    # Plot final losses
    loss_plot_path = os.path.join(result_folder, "losses_final.png")
    plot_losses(epoch_losses, save_path=loss_plot_path)

    total_time = time.perf_counter() - start_time_total
    print(f"\n" + "=" * 70)
    print(f"TRAINING COMPLETE")
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    print("=" * 70)

    print(f"\nTotal training time: {total_time:.2f}s ({total_time / 60:.2f} minutes)\n")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../../data/processed")
    ap.add_argument("--checkpoint_path", type=str, default=None)
    ap.add_argument("--pretrained_model", type=str, default=None)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--test_split", type=float, default=0.1)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--label_dim", type=int, default=512)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--n_sample", type=int, default=2)
    args = ap.parse_args()

    train(args)

    print("\n" + "=" * 70)
    print("GENERATING SAMPLE IMAGES")
    print("=" * 70)

    sample_prompts = [
        "Delicious macaroni in a rich tomato sauce with crispy bacon, infused with herbs and spices for the perfect comfort meal. Enjoy a bowl of warmth and flavor!",
        "Savor the rich flavors of sautéed beef liver, garlic, and onions, finished with a touch of egg and fresh parsley for a comforting and hearty dish. Perfectly cooked and served warm, this Juneća džigerica is a delectable treat!"
    ]

    # generate_sample_images(sample_prompts,model=model, num_images=10,save_folder=args.out_dir)

    print("\n" + "=" * 70)
    print(f"Results saved to: {args.out_dir}")
    print("=" * 70)
