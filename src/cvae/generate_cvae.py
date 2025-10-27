import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

from model import ConvCVAE, Decoder, Encoder
from image_generation import encode_text
from utils import convert_batch_to_image_grid


def load_model(args):
    encoder = Encoder(args.latent_dim, concat_input_and_condition=True, image_size=args.img_size)
    decoder = Decoder(batch_size=args.batch, image_size=args.img_size)
    model = ConvCVAE(
        encoder,
        decoder,
        label_dim=args.label_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        batch_size=args.batch,
        image_dim=[args.img_size, args.img_size, 3]
    )
    checkpoint_root = os.path.join(".", "checkpoints", args.checkpoint_name)
    checkpoint = tf.train.Checkpoint(module=model)
    latest = tf.train.latest_checkpoint(checkpoint_root)

    if latest is None:
        print(f"ERROR: No checkpoint found in {checkpoint_root}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = os.path.join(".", "checkpoints")
        if os.path.exists(checkpoint_dir):
            for folder in os.listdir(checkpoint_dir):
                folder_path = os.path.join(checkpoint_dir, folder)
                if os.path.isdir(folder_path):
                    print(f"  - {folder}")
        sys.exit(1)

    checkpoint.restore(latest).expect_partial()
    print(f"✓ Checkpoint loaded: {latest}\n")

    return model


def generate_images(model, prompt, num_images, save_path):
    print(f"Generating {num_images} images for: '{prompt}'")

    # Get text embedding
    text_embedding = encode_text(prompt)
    condition = np.tile(text_embedding, (num_images, 1))
    condition_tf = tf.constant(condition, dtype=tf.float32)

    # Generate
    generated = model.generate(condition_tf)
    generated_np = generated.numpy()

    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(convert_batch_to_image_grid(generated_np))
    plt.axis('off')
    plt.title(prompt, fontsize=16, pad=20)

    # Save
    prompt_clean = prompt.replace(' ', '_').replace('/', '-')[:50]
    save_file = os.path.join(save_path, f"generation_{prompt_clean}.png")
    plt.savefig(save_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved to: {save_file}\n")

    if args.show:
        plt.show()
    plt.close()

    return generated_np

def generate_variations(model, prompt, num_variations, save_path):
    """Generate variations of the same prompt."""
    print(f"Generating {num_variations} variations for: '{prompt}'")

    # Get text embedding
    text_embedding = encode_text(prompt)

    # Generate multiple times with different random seeds
    all_images = []
    for i in range(num_variations):
        condition_tf = tf.constant([text_embedding], dtype=tf.float32)
        generated = model.generate(condition_tf, num_samples=1)
        all_images.append(generated.numpy()[0])

    all_images = np.array(all_images)

    # Calculate grid size
    cols = min(4, num_variations)
    rows = (num_variations + cols - 1) // cols

    # Plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if num_variations == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows * cols > 1 else [axes]

    for i in range(num_variations):
        axes[i].imshow(all_images[i])
        axes[i].axis('off')
        axes[i].set_title(f"Variation {i + 1}")

    # Hide unused subplots
    for i in range(num_variations, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Variations: {prompt}', fontsize=16)
    plt.tight_layout()

    # Save
    prompt_clean = prompt.replace(' ', '_').replace('/', '-')[:50]
    save_file = os.path.join(save_path, f"variations_{prompt_clean}.png")
    plt.savefig(save_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved to: {save_file}\n")

    if args.show:
        plt.show()
    plt.close()

    return all_images


def interpolate_between_prompts(model, prompt1, prompt2, num_steps, save_path):
    """Generate images interpolating between two prompts."""
    print(f"Interpolating between:")
    print(f"  '{prompt1}' -> '{prompt2}'")

    # Get text embeddings
    embedding1 = encode_text(prompt1)
    embedding2 = encode_text(prompt2)

    # Create interpolation steps
    alphas = np.linspace(0, 1, num_steps)
    interpolated_images = []

    for alpha in alphas:
        # Interpolate embeddings
        interpolated_embedding = (1 - alpha) * embedding1 + alpha * embedding2

        # Generate image
        condition_tf = tf.constant([interpolated_embedding], dtype=tf.float32)
        generated = model.generate(condition_tf, num_samples=1)
        interpolated_images.append(generated.numpy()[0])

    interpolated_images = np.array(interpolated_images)

    # Plot
    cols = min(8, num_steps)
    rows = (num_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if num_steps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows * cols > 1 else [axes]

    for i in range(num_steps):
        axes[i].imshow(interpolated_images[i])
        axes[i].axis('off')
        axes[i].set_title(f"{alphas[i]:.2f}", fontsize=10)

    # Hide unused subplots
    for i in range(num_steps, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'"{prompt1}" -> "{prompt2}"', fontsize=14)
    plt.tight_layout()

    # Save
    prompt1_clean = prompt1.replace(' ', '_')[:20]
    prompt2_clean = prompt2.replace(' ', '_')[:20]
    save_file = os.path.join(save_path, f"interpolation_{prompt1_clean}_to_{prompt2_clean}.png")
    plt.savefig(save_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved to: {save_file}\n")

    if args.show:
        plt.show()
    plt.close()

    return interpolated_images


def generate_from_file(model, prompt_file, num_images, save_path):
    """Generate images from prompts in a text file."""
    print(f"Reading prompts from: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Found {len(prompts)} prompts\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}]")
        generate_images(model, prompt, num_images, save_path)


def main(args):
    # Create output directory
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_folder = os.path.join(args.out_dir, f"generation_{timestamp_str}")
    os.makedirs(session_folder, exist_ok=True)

    print(f"Output folder: {session_folder}\n")

    # Load model
    model = load_model(args)

    print("=" * 70)
    print("GENERATING IMAGES")
    print("=" * 70 + "\n")

    # Generate based on input type
    if args.interpolate:
        if len(args.interpolate) != 2:
            print("ERROR: --interpolate requires exactly 2 prompts")
            sys.exit(1)
        interpolate_between_prompts(
            model,
            args.interpolate[0],
            args.interpolate[1],
            args.num_steps,
            session_folder
        )

    elif args.variations:
        if not args.prompt:
            print("ERROR: --variations requires --prompt")
            sys.exit(1)
        generate_variations(model, args.prompt, args.variations, session_folder)

    elif args.prompt_file:
        generate_from_file(model, args.prompt_file, args.num_images, session_folder)

    elif args.prompts:
        for i, prompt in enumerate(args.prompts, 1):
            print(f"[{i}/{len(args.prompts)}]")
            generate_images(model, prompt, args.num_images, session_folder)

    elif args.prompt:
        generate_images(model, args.prompt, args.num_images, session_folder)

    else:
        print("ERROR: No prompt provided. Use --prompt, --prompts, or --prompt_file")
        sys.exit(1)

    print("=" * 70)
    print("GENERATION COMPLETE")
    print(f"All images saved to: {session_folder}")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate images from trained cVAE model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt
  python generate.py --prompt "a traditional Serbian dish"

  # Multiple prompts
  python generate.py --prompts "grilled meat" "cheese pastry" "soup"

  # From file
  python generate.py --prompt_file prompts.txt

  # Variations
  python generate.py --prompt "Serbian bread" --variations 8

  # Interpolation
  python generate.py --interpolate "grilled meat" "baked pastry"
        """
    )

    # Input options (mutually exclusive)
    input_group = ap.add_mutually_exclusive_group()
    input_group.add_argument("--prompt", type=str, help="Single text prompt")
    input_group.add_argument("--prompts", type=str, nargs='+', help="Multiple text prompts")
    input_group.add_argument("--prompt_file", type=str, help="File with prompts (one per line)")

    # Special generation modes
    ap.add_argument("--variations", type=int, help="Generate N variations of --prompt")
    ap.add_argument("--interpolate", type=str, nargs=2, metavar=('PROMPT1', 'PROMPT2'),
                    help="Interpolate between two prompts")
    ap.add_argument("--num_steps", type=int, default=8,
                    help="Number of interpolation steps (default: 8)")

    # Generation settings
    ap.add_argument("--num_images", type=int, default=32,
                    help="Number of images to generate per prompt (default: 32)")
    ap.add_argument("--show", action="store_true",
                    help="Display images in addition to saving them")

    # Model settings (must match trained model)
    ap.add_argument("--checkpoint_name", type=str, required=True,
                    help="Checkpoint folder name (e.g., 2025-01-15_14-30-00)")
    ap.add_argument("--img_size", type=int, default=128,
                    help="Image size (default: 128)")
    ap.add_argument("--latent_dim", type=int, default=128,
                    help="Latent dimension (default: 128)")
    ap.add_argument("--label_dim", type=int, default=512,
                    help="Label/embedding dimension (default: 512)")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="Beta parameter (default: 1.0)")
    ap.add_argument("--batch", type=int, default=32,
                    help="Batch size (default: 32)")

    # Output settings
    ap.add_argument("--out_dir", type=str, default="generated_images",
                    help="Output directory (default: generated_images)")

    args = ap.parse_args()

    main(args)