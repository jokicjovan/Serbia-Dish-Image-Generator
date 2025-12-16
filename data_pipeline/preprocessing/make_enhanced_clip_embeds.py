import argparse
import os
from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
from tqdm import tqdm

try:
    import open_clip
    import torch
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError(
        "This script needs 'open_clip_torch' and 'torch'. Install with:\n"
        "  pip install open_clip_torch torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121\n"
        "Or use the CPU-only wheel if you don't have CUDA."
    ) from e

# Food-specific prompt templates for better conditioning
FOOD_TEMPLATES = [
    "a photo of {text}",
    "a delicious {text}",
    "traditional {text}",
    "homemade {text}",
    "fresh {text}",
    "authentic {text}",
    "a plate of {text}",
    "{text} on a white plate"
]

def list_caption_files(captions_dir: Path) -> List[Path]:
    files = sorted([p for p in captions_dir.glob("*.txt") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .txt captions found in: {captions_dir}")
    return files

def load_texts(paths: List[Path]) -> List[str]:
    texts = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8").strip()
            # Clean and normalize text
            if not text or text == " ":
                text = "food on plate"  # fallback
            texts.append(text)
        except UnicodeDecodeError:
            texts.append(p.read_text(encoding="latin-1").strip())
    return texts

def enhance_text_with_templates(text: str, use_templates: bool = True) -> List[str]:
    """Generate multiple variations of text for ensemble encoding."""
    if not use_templates:
        return [text]

    # Clean text first
    text = text.strip().lower()

    # Generate variations using templates
    variations = []
    for template in FOOD_TEMPLATES:
        variations.append(template.format(text=text))

    # Also include original
    variations.append(text)

    return variations

def encode_text_ensemble(model, tokenizer, texts: List[str], device: str,
                        ensemble_method: str = "mean") -> torch.Tensor:
    """Encode multiple text variations and combine them."""
    all_features = []

    for text in texts:
        tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features)

    # Combine features
    if ensemble_method == "mean":
        combined = torch.stack(all_features).mean(dim=0)
    elif ensemble_method == "max":
        combined = torch.stack(all_features).max(dim=0)[0]
    elif ensemble_method == "weighted":
        # Give more weight to original text
        weights = torch.tensor([0.3] * (len(all_features) - 1) + [0.4]).to(device)
        weights = weights / weights.sum()
        combined = torch.stack(all_features).T.matmul(weights).T
    else:
        combined = torch.stack(all_features).mean(dim=0)

    # Re-normalize
    combined = combined / combined.norm(dim=-1, keepdim=True)
    return combined

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def infer_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return name

def get_model_info(model_name: str) -> Tuple[str, int]:
    """Get model info including embedding dimension."""
    model_dims = {
        "ViT-B-32": 512,
        "ViT-B-16": 512,
        "ViT-L-14": 768,
        "ViT-L-14-336": 768,
        "ViT-H-14": 1024,
        "ViT-g-14": 1024,
        "ViT-bigG-14": 1280,
    }

    dim = model_dims.get(model_name, 512)
    return model_name, dim

def main():
    ap = argparse.ArgumentParser(description="Generate enhanced CLIP embeddings for better food conditioning")

    # I/O
    ap.add_argument("--captions_dir", type=str, default="data/processed/captions",
                   help="Folder with <image_id>.txt")
    ap.add_argument("--embeds_dir", type=str, default="data/processed/embeddings_enhanced",
                   help="Output folder for <image_id>.npy")

    # Model selection
    ap.add_argument("--model", type=str, default="ViT-L-14",
                   help="OpenCLIP model (ViT-B-32: 512dim, ViT-L-14: 768dim, ViT-H-14: 1024dim)")
    ap.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k",
                   help="OpenCLIP pretrained dataset")

    # Enhancement options
    ap.add_argument("--use_ensemble", action="store_true",
                   help="Use multiple text templates for ensemble encoding")
    ap.add_argument("--ensemble_method", type=str, default="weighted",
                   choices=["mean", "max", "weighted"], help="How to combine ensemble features")

    # Processing
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    ap.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'mps', or 'cpu'")

    # Verification
    ap.add_argument("--save_metadata", action="store_true",
                   help="Save embedding metadata (model, dim, etc.)")

    args = ap.parse_args()

    captions_dir = Path(args.captions_dir)
    embeds_dir = Path(args.embeds_dir)
    ensure_dir(embeds_dir)

    files = list_caption_files(captions_dir)
    texts = load_texts(files)

    device = infer_device(args.device)
    model_name, embed_dim = get_model_info(args.model)

    print(f"📊 Processing {len(files)} captions")
    print(f"🤖 Model: {args.model} ({embed_dim}D) on {device}")
    print(f"🔄 Ensemble: {'Yes' if args.use_ensemble else 'No'}")
    if args.use_ensemble:
        print(f"🎯 Ensemble method: {args.ensemble_method}")
    print(f"💾 Output: {embeds_dir}")
    print()

    print(f"Loading OpenCLIP model: {args.model} ({args.pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    print(f"✅ Model loaded successfully (embedding dim: {embed_dim})")

    # Save metadata
    if args.save_metadata:
        metadata = {
            "model": args.model,
            "pretrained": args.pretrained,
            "embedding_dim": embed_dim,
            "ensemble": args.use_ensemble,
            "ensemble_method": args.ensemble_method if args.use_ensemble else None,
            "num_samples": len(files)
        }

        with open(embeds_dir / "embedding_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Saved metadata to {embeds_dir / 'embedding_metadata.json'}")

    # Process embeddings
    with torch.no_grad():
        for i in tqdm(range(0, len(files), args.batch_size), desc="🔄 Encoding"):
            batch_paths = files[i:i+args.batch_size]
            batch_texts = texts[i:i+args.batch_size]

            for path, text in zip(batch_paths, batch_texts):
                # Enhance text if using ensemble
                if args.use_ensemble:
                    text_variations = enhance_text_with_templates(text, True)
                    features = encode_text_ensemble(
                        model, tokenizer, text_variations, device, args.ensemble_method
                    )
                else:
                    tokens = tokenizer([text]).to(device)
                    features = model.encode_text(tokens)
                    features = features / features.norm(dim=-1, keepdim=True)

                # Save embedding
                embedding = features.squeeze(0).detach().cpu().numpy().astype("float32")
                out_path = embeds_dir / f"{path.stem}.npy"
                np.save(out_path, embedding, allow_pickle=False)

    print(f"\n✅ Generated {len(files)} enhanced embeddings")
    print(f"📁 Output directory: {embeds_dir}")
    print(f"📏 Embedding dimension: {embed_dim}")

    if args.use_ensemble:
        print(f"🎯 Used ensemble method: {args.ensemble_method}")

    print("\n🔄 Next steps:")
    print(f"1. Update your training script to use embeddings from: {embeds_dir}")
    print(f"2. Update model architecture for {embed_dim}-dim embeddings")
    if embed_dim != 512:
        print(f"3. Set --cond_dim to at least {embed_dim} in training")

if __name__ == "__main__":
    main()