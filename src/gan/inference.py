import os, argparse, torch, glob
import numpy as np
from PIL import Image
from torchvision.utils import save_image, make_grid
from models import Generator

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
        # Fallback: infer from generator's embed layer weight shape
        gen_state = ckpt['G']
        if 'embed.0.weight' in gen_state:
            cond_in = gen_state['embed.0.weight'].shape[1]
        else:
            print("WARNING: Could not determine embedding dimension from checkpoint.")
            print("Defaulting to 512. If inference fails, check your embedding files.")
            cond_in = 512

    print(f"Using embedding dimension: {cond_in}")

    G = Generator(
        z_dim=args.get('z_dim', 128),
        cond_in=cond_in,
        cond_hidden=args.get('cond_dim', 256),
        base_ch=args.get('base_ch', 64),
        out_size=args.get('img_size', 128)
    ).to(device)
    
    # Load EMA weights if available, otherwise regular weights
    if 'ema' in ckpt:
        print("Loading EMA weights")
        # First load regular weights to get BN stats
        G.load_state_dict(ckpt['G'])
        # Then apply EMA shadow weights
        from ema import EMA
        ema = EMA(G, decay=0.999)
        ema.shadow = ckpt['ema']
        ema.apply_shadow(G)
    else:
        print("Loading regular generator weights")
        G.load_state_dict(ckpt['G'])
    
    G.eval()
    return G, args

def load_embedding(emb_path):
    """Load a text embedding from .npy file."""
    emb = np.load(emb_path).astype('float32')
    return torch.from_numpy(emb)

def generate_from_embedding(G, embedding, z_dim=128, num_samples=1, device='cuda'):
    """Generate images from a single embedding."""
    with torch.no_grad():
        # Repeat embedding for batch
        e = embedding.unsqueeze(0).repeat(num_samples, 1).to(device)
        
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

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint, device)
    G, train_args = create_generator(ckpt, device)
    z_dim = train_args.get('z_dim', 128)
    
    print(f"Generator loaded (z_dim={z_dim}, img_size={train_args.get('img_size', 128)})")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Mode 1: Generate from specific embedding file
    if args.embedding:
        print(f"\nGenerating from embedding: {args.embedding}")
        embedding = load_embedding(args.embedding)
        imgs = generate_from_embedding(G, embedding, z_dim, args.num_samples, device)
        
        if args.save_grid:
            save_samples(imgs, os.path.join(args.output_dir, "generated_grid.png"), 
                        nrow=int(np.sqrt(args.num_samples)))
        
        if args.save_individual:
            save_individual(imgs, args.output_dir, prefix="sample")
    
    # Mode 2: Generate from all embeddings in directory
    elif args.embedding_dir:
        print(f"\nGenerating from all embeddings in: {args.embedding_dir}")
        emb_files = sorted(glob.glob(os.path.join(args.embedding_dir, "*.npy")))
        print(f"Found {len(emb_files)} embedding files")
        
        for emb_file in emb_files:
            base_name = os.path.splitext(os.path.basename(emb_file))[0]
            print(f"Processing {base_name}...")
            
            embedding = load_embedding(emb_file)
            imgs = generate_from_embedding(G, embedding, z_dim, args.num_samples, device)
            
            if args.save_grid:
                save_samples(imgs, 
                           os.path.join(args.output_dir, f"{base_name}_grid.png"),
                           nrow=int(np.sqrt(args.num_samples)))
            
            if args.save_individual:
                img_dir = os.path.join(args.output_dir, base_name)
                save_individual(imgs, img_dir, prefix="sample")
    
    # Mode 3: Random generation with random embeddings
    else:
        print(f"\nGenerating {args.num_samples} random images...")
        # Create random embeddings with correct dimension
        cond_in = ckpt['args'].get('cond_in') if ckpt['args'].get('cond_in') else 512
        random_embeddings = torch.randn(args.num_samples, cond_in, device=device)
        z = torch.randn(args.num_samples, z_dim, device=device)
        
        with torch.no_grad():
            imgs = G(z, random_embeddings)
        
        if args.save_grid:
            save_samples(imgs, os.path.join(args.output_dir, "random_grid.png"),
                        nrow=int(np.sqrt(args.num_samples)))
        
        if args.save_individual:
            save_individual(imgs, args.output_dir, prefix="random")
    
    print(f"\nDone! Check {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from trained conditional GAN")
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file (e.g., runs/cgan/ckpt_0050000.pt)")
    
    # Input modes (choose one)
    parser.add_argument("--embedding", type=str, default=None,
                       help="Path to single .npy embedding file")
    parser.add_argument("--embedding_dir", type=str, default=None,
                       help="Directory containing multiple .npy embedding files")
    
    # Generation settings
    parser.add_argument("--num_samples", type=int, default=16,
                       help="Number of samples to generate per embedding")
    parser.add_argument("--output_dir", type=str, default="generated",
                       help="Output directory for generated images")
    
    # Output format
    parser.add_argument("--save_grid", action="store_true", default=True,
                       help="Save images as grid")
    parser.add_argument("--save_individual", action="store_true",
                       help="Save individual images")
    parser.add_argument("--no_grid", dest='save_grid', action="store_false",
                       help="Don't save grid (only individuals)")
    
    args = parser.parse_args()
    
    # Validate
    if args.embedding and args.embedding_dir:
        parser.error("Cannot specify both --embedding and --embedding_dir")
    
    main(args)