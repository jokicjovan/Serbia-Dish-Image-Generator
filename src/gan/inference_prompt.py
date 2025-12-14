import os, argparse, torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image, make_grid
from models import Generator
from ema import EMA

def load_checkpoint(ckpt_path, device):
    """Load checkpoint and extract model state and args."""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt

def create_generator(ckpt, device):
    """Create generator from checkpoint."""
    args = ckpt['args']

    # Get embedding dimension from checkpoint args or generator state dict
    cond_in = args.get('cond_in')  # Try to get from saved args first
    if cond_in is None:
        # Fallback: infer from generator's conditioning layer weight shape
        gen_state = ckpt['G']
        if 'cond.0.weight' in gen_state:
            # The conditioning layer input size tells us the embedding dimension
            cond_in = gen_state['cond.0.weight'].shape[1]
            print(f"Detected embedding dimension from cond.0.weight: {cond_in}")
        elif 'embed.0.weight' in gen_state:
            cond_in = gen_state['embed.0.weight'].shape[1]
            print(f"Detected embedding dimension from embed.0.weight: {cond_in}")
        else:
            print("WARNING: Could not determine embedding dimension from checkpoint.")
            print("Analyzing generator state dict keys...")
            # Print available keys to help debug
            cond_keys = [k for k in gen_state.keys() if 'cond' in k or 'embed' in k]
            print(f"Available conditioning keys: {cond_keys}")
            print("Note: CLIP ViT-B/32 produces 512-dim embeddings, ViT-L/14 produces 768-dim")
            cond_in = 512  # Default to CLIP ViT-B/32

    print(f"Using embedding dimension: {cond_in}")

    # Determine CLIP model based on embedding dimension
    if cond_in <= 512:
        clip_model_name = "ViT-B/32"
        expected_dim = 512
    elif cond_in <= 768:
        clip_model_name = "ViT-L/14"
        expected_dim = 768
    elif cond_in <= 1024:
        clip_model_name = "ViT-H/14"
        expected_dim = 1024
    else:
        clip_model_name = "ViT-B/32"  # fallback
        expected_dim = 512
        print(f"WARNING: Unusual embedding dim {cond_in}, using ViT-B/32 as fallback")

    print(f"Recommended CLIP model: {clip_model_name} (produces {expected_dim}D embeddings)")

    G = Generator(
        z_dim=args.get('z_dim', 128),
        cond_in=cond_in,
        cond_hidden=args.get('cond_dim', 256),
        base_ch=args.get('base_ch', 64),
        out_size=args.get('img_size', 128)
    ).to(device)

    # Load EMA weights if available
    if 'ema' in ckpt:
        print("Loading EMA weights")
        G.load_state_dict(ckpt['G'])
        ema = EMA(G, decay=0.999)
        ema.shadow = ckpt['ema']
        ema.apply_shadow(G)
    else:
        print("Loading regular generator weights")
        G.load_state_dict(ckpt['G'])

    G.eval()
    return G, args, clip_model_name

def load_clip_model(device, model_name="ViT-B/32"):
    """Load CLIP model for text encoding."""
    try:
        import open_clip
    except ImportError:
        print("ERROR: open_clip_torch not installed!")
        print("Install with: pip install open_clip_torch")
        exit(1)

    # Map model names to correct pretrained weights
    model_pretraining_map = {
        "ViT-B/32": "laion2b_s34b_b79k",
        "ViT-L/14": "laion2b_s32b_b82k",  # This is the correct one for your model!
        "ViT-H/14": "laion2b_s32b_b79k",
    }

    pretrained = model_pretraining_map.get(model_name, "laion2b_s34b_b79k")

    print(f"Loading CLIP model: {model_name} ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, tokenizer

def encode_text(clip_model, tokenizer, text, device, target_dim=None):
    """Encode text prompt to CLIP embedding."""
    import open_clip

    # Tokenize and encode
    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        # Normalize (CLIP outputs normalized features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Convert to float32 to match generator
    embedding = text_features.squeeze(0).float()

    # Check dimension compatibility
    if target_dim is not None and embedding.shape[0] != target_dim:
        print(f"WARNING: CLIP embedding dim ({embedding.shape[0]}) != target dim ({target_dim})")
        if embedding.shape[0] < target_dim:
            # Pad with zeros if CLIP embedding is smaller
            padding = torch.zeros(target_dim - embedding.shape[0], device=device)
            embedding = torch.cat([embedding, padding])
            print(f"Padded embedding to {target_dim}D")
        else:
            # Truncate if CLIP embedding is larger
            embedding = embedding[:target_dim]
            print(f"Truncated embedding to {target_dim}D")

    return embedding

def generate_from_prompt(G, clip_model, tokenizer, prompt, z_dim=128, num_samples=1, device='cuda', target_dim=None):
    """Generate images from text prompt."""
    print(f"\nPrompt: '{prompt}'")

    # Encode prompt
    embedding = encode_text(clip_model, tokenizer, prompt, device, target_dim)

    with torch.no_grad():
        # Repeat embedding for batch
        e = embedding.unsqueeze(0).repeat(num_samples, 1)

        # Sample random noise
        z = torch.randn(num_samples, z_dim, device=device)

        # Generate
        imgs = G(z, e)

    return imgs

def save_samples(imgs, output_path, nrow=4):
    """Save generated images as grid."""
    grid = make_grid((imgs.clamp(-1,1)+1)/2, nrow=nrow)
    save_image(grid, output_path)
    print(f"Saved samples to {output_path}")

def save_individual(imgs, output_dir, prefix="sample"):
    """Save individual images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        img_normalized = (img.clamp(-1, 1) + 1) / 2
        save_image(img_normalized, os.path.join(output_dir, f"{prefix}_{i:04d}.png"))
    print(f"Saved {len(imgs)} individual images to {output_dir}")

def interactive_mode(G, clip_model, tokenizer, z_dim, output_dir, device, cond_in):
    """Interactive prompt entry mode."""
    print("\n" + "="*60)
    print("Interactive Generation Mode")
    print("="*60)
    print("Enter prompts to generate images. Type 'quit' or 'exit' to stop.")
    print("Type 'help' for options.\n")

    prompt_count = 0

    while True:
        try:
            prompt = input("Prompt > ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if prompt.lower() == 'help':
                print("\nCommands:")
                print("  - Enter any text to generate images")
                print("  - 'quit' or 'exit' to stop")
                print("  - 'help' for this message\n")
                continue

            # Generate images
            imgs = generate_from_prompt(G, clip_model, tokenizer, prompt, z_dim,
                                       num_samples=16, device=device, target_dim=cond_in)

            # Save with sanitized filename
            safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_'
                               for c in prompt)[:50]
            output_path = os.path.join(output_dir, f"{prompt_count:04d}_{safe_name}.png")

            save_samples(imgs, output_path, nrow=4)
            prompt_count += 1
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    ckpt = load_checkpoint(args.checkpoint, device)
    G, train_args, clip_model_name = create_generator(ckpt, device)
    z_dim = train_args.get('z_dim', 128)

    # Get embedding dimension from training args or detect from model
    cond_in = train_args.get('cond_in')
    if cond_in is None:
        # Try to get from generator state if not in args
        gen_state = ckpt['G']
        if 'cond.0.weight' in gen_state:
            cond_in = gen_state['cond.0.weight'].shape[1]
        else:
            cond_in = 512  # fallback

    clip_model, tokenizer = load_clip_model(device, clip_model_name)

    print(f"Generator loaded (z_dim={z_dim}, embedding_dim={cond_in}, img_size={train_args.get('img_size', 128)})")
    print("CLIP model loaded\n")

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Interactive mode
    if args.interactive:
        interactive_mode(G, clip_model, tokenizer, z_dim, args.output_dir, device, cond_in)

    # Single prompt mode
    elif args.prompt:
        imgs = generate_from_prompt(G, clip_model, tokenizer, args.prompt, z_dim,
                                    args.num_samples, device, cond_in)

        if args.save_grid:
            save_samples(imgs,
                        os.path.join(args.output_dir, "generated_grid.png"),
                        nrow=int(np.sqrt(args.num_samples)))

        if args.save_individual:
            save_individual(imgs, args.output_dir, prefix="sample")

    # Prompts from file
    elif args.prompts_file:
        print(f"Reading prompts from {args.prompts_file}")
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"Found {len(prompts)} prompts\n")

        for i, prompt in enumerate(prompts):
            imgs = generate_from_prompt(G, clip_model, tokenizer, prompt, z_dim,
                                       args.num_samples, device, cond_in)

            safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_'
                               for c in prompt)[:50]

            if args.save_grid:
                output_path = os.path.join(args.output_dir,
                                          f"{i:04d}_{safe_name}_grid.png")
                save_samples(imgs, output_path,
                           nrow=int(np.sqrt(args.num_samples)))

            if args.save_individual:
                img_dir = os.path.join(args.output_dir, f"{i:04d}_{safe_name}")
                save_individual(imgs, img_dir, prefix="sample")

    else:
        print("ERROR: Must specify --prompt, --prompts_file, or --interactive")
        print("Run with --help for usage information")

    print(f"\nDone! Check {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using trained conditional GAN + CLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt
  python inference_prompt.py --checkpoint runs/cgan/ckpt_0050000.pt --prompt "a delicious pizza"
  
  # Interactive mode
  python inference_prompt.py --checkpoint runs/cgan/ckpt_0050000.pt --interactive
  
  # Multiple prompts from file
  python inference_prompt.py --checkpoint runs/cgan/ckpt_0050000.pt --prompts_file prompts.txt
        """)
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    
    # Input modes (choose one)
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single text prompt")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="Text file with one prompt per line")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive prompt entry mode")
    
    # Generation settings
    parser.add_argument("--num_samples", type=int, default=16,
                       help="Number of samples to generate per prompt")
    parser.add_argument("--output_dir", type=str, default="generated_prompts",
                       help="Output directory for generated images")
    
    # Output format
    parser.add_argument("--save_grid", action="store_true", default=True,
                       help="Save images as grid")
    parser.add_argument("--save_individual", action="store_true",
                       help="Save individual images")
    parser.add_argument("--no_grid", dest='save_grid', action="store_false",
                       help="Don't save grid (only individuals)")
    
    args = parser.parse_args()
    main(args)