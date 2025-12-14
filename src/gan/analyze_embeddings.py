import os
import json
import argparse
import numpy as np
from pathlib import Path
import torch

try:
    import open_clip
except ImportError:
    print("ERROR: open_clip not installed!")
    print("Install with: pip install open_clip_torch")
    exit(1)

def analyze_training_embeddings(embeddings_dir):
    """Analyze training embeddings to determine generation method."""
    print("🔍 Analyzing training embeddings...")

    emb_files = list(Path(embeddings_dir).glob("*.npy"))
    if not emb_files:
        print(f"No .npy files found in {embeddings_dir}")
        return None

    # Load sample embeddings
    sample_embeddings = []
    for i in range(min(10, len(emb_files))):
        emb = np.load(emb_files[i])
        sample_embeddings.append(emb)

    sample_embeddings = np.stack(sample_embeddings)

    print(f"📊 Training embedding analysis:")
    print(f"  Shape: {sample_embeddings[0].shape}")
    print(f"  Mean: {sample_embeddings.mean():.4f}")
    print(f"  Std: {sample_embeddings.std():.4f}")
    print(f"  Min: {sample_embeddings.min():.4f}")
    print(f"  Max: {sample_embeddings.max():.4f}")
    print(f"  Norm range: {np.linalg.norm(sample_embeddings, axis=1).min():.4f} - {np.linalg.norm(sample_embeddings, axis=1).max():.4f}")

    return {
        'dim': sample_embeddings.shape[1],
        'mean': float(sample_embeddings.mean()),
        'std': float(sample_embeddings.std()),
        'min': float(sample_embeddings.min()),
        'max': float(sample_embeddings.max()),
        'norm_mean': float(np.linalg.norm(sample_embeddings, axis=1).mean()),
        'norm_std': float(np.linalg.norm(sample_embeddings, axis=1).std()),
    }

def test_clip_models(text_sample, target_stats, device):
    """Test different CLIP models to find closest match."""
    print("🤖 Testing different CLIP models...")

    models_to_test = [
        ("ViT-B-32", "laion2b_s34b_b79k"),
        ("ViT-B-32", "openai"),
        ("ViT-B-16", "laion2b_s34b_b88k"),
        ("ViT-L-14", "laion2b_s32b_b82k"),
        ("ViT-L-14", "openai"),
    ]

    best_match = None
    best_similarity = -1
    results = []

    for model_name, pretrained in models_to_test:
        try:
            print(f"  Testing {model_name} ({pretrained})...")
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
            tokenizer = open_clip.get_tokenizer(model_name)

            tokens = tokenizer([text_sample]).to(device)
            with torch.no_grad():
                embedding = model.encode_text(tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            emb_np = embedding.squeeze().cpu().numpy()

            # Calculate statistics
            stats = {
                'model': f"{model_name}_{pretrained}",
                'dim': emb_np.shape[0],
                'mean': float(emb_np.mean()),
                'std': float(emb_np.std()),
                'norm': float(np.linalg.norm(emb_np))
            }

            # Score based on similarity to training stats
            if stats['dim'] == target_stats['dim']:
                mean_diff = abs(stats['mean'] - target_stats['mean'])
                std_diff = abs(stats['std'] - target_stats['std'])
                norm_diff = abs(stats['norm'] - target_stats['norm_mean'])

                # Simple similarity score (lower is better)
                similarity = -(mean_diff + std_diff + norm_diff)

                print(f"    Dim: {stats['dim']}, Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Norm: {stats['norm']:.4f}")
                print(f"    Similarity score: {similarity:.4f}")

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (model_name, pretrained)

                stats['similarity'] = similarity
            else:
                print(f"    Dimension mismatch: {stats['dim']} != {target_stats['dim']}")
                stats['similarity'] = -999

            results.append(stats)

        except Exception as e:
            print(f"    Failed to load {model_name} ({pretrained}): {e}")

    return best_match, results

def create_matched_embeddings(text_list, model_name, pretrained, device):
    """Generate embeddings using the matched model."""
    print(f"🎯 Generating embeddings with {model_name} ({pretrained})...")

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    embeddings = []

    for text in text_list:
        tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            embedding = model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embeddings.append(embedding.squeeze().cpu().numpy())

    return np.array(embeddings)

def main():
    parser = argparse.ArgumentParser(description="Analyze training embeddings and find matching CLIP model")

    parser.add_argument("--embeddings_dir", type=str, default="data/processed/embedds",
                       help="Directory with training embeddings")
    parser.add_argument("--captions_dir", type=str, default="data/processed/captions",
                       help="Directory with training captions")
    parser.add_argument("--output_file", type=str, default="embedding_analysis.json",
                       help="Output file for analysis results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    print(f"Using device: {device}")

    # Analyze training embeddings
    embedding_stats = analyze_training_embeddings(args.embeddings_dir)
    if embedding_stats is None:
        return

    # Get a sample caption for testing
    caption_files = list(Path(args.captions_dir).glob("*.txt"))
    if not caption_files:
        print(f"No caption files found in {args.captions_dir}")
        text_sample = "traditional Serbian food"
    else:
        with open(caption_files[0], 'r', encoding='utf-8') as f:
            text_sample = f.read().strip()

    print(f"📝 Using sample text: '{text_sample}'")

    # Test different CLIP models
    best_match, results = test_clip_models(text_sample, embedding_stats, device)

    # Save analysis results
    analysis_results = {
        'training_stats': embedding_stats,
        'sample_text': text_sample,
        'model_test_results': results,
        'best_match': best_match,
        'device': device
    }

    with open(args.output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n📋 Analysis complete!")
    print(f"📊 Results saved to: {args.output_file}")

    if best_match:
        model_name, pretrained = best_match
        print(f"🎯 Best matching model: {model_name} ({pretrained})")
        print(f"\n📝 Update your inference script to use:")
        print(f"   Model: {model_name}")
        print(f"   Pretrained: {pretrained}")
    else:
        print("❌ No good match found. Your embeddings might be from a custom model.")

if __name__ == "__main__":
    main()