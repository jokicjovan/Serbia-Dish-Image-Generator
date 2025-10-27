import argparse
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

try:
    import open_clip
    import torch
except Exception as e:
    raise RuntimeError(
        "This script needs 'open_clip_torch' and 'torch'. Install with:\n"
        "  pip install open_clip_torch torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121\n"
        "Or use the CPU-only wheel if you don't have CUDA."
    ) from e

def list_caption_files(captions_dir: Path) -> List[Path]:
    files = sorted([p for p in captions_dir.glob("*.txt") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .txt captions found in: {captions_dir}")
    return files

def load_texts(paths: List[Path]) -> List[str]:
    texts = []
    for p in paths:
        try:
            texts.append(p.read_text(encoding="utf-8").strip())
        except UnicodeDecodeError:
            texts.append(p.read_text(encoding="latin-1").strip())
    return texts

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def infer_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_dir", type=str, default="refined_data/captions", help="Folder with <image_id>.txt")
    ap.add_argument("--embeds_dir", type=str, default="refined_data/embeds", help="Output folder for <image_id>.npy")
    ap.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name (e.g., ViT-B-32, ViT-L-14)")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    ap.add_argument("--device", type=str, default="auto", help="'auto', 'cuda' or 'cpu'")
    args = ap.parse_args()

    captions_dir = Path(args.captions_dir)
    embeds_dir = Path(args.embeds_dir)
    ensure_dir(embeds_dir)

    files = list_caption_files(captions_dir)
    texts = load_texts(files)

    device = infer_device(args.device)

    print(f"Loading OpenCLIP model: {args.model} ({args.pretrained}) on {device} ...")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(files), args.batch_size), desc="Encoding captions"):
            batch_paths = files[i:i+args.batch_size]
            batch_texts = [t if t else " " for t in texts[i:i+args.batch_size]]

            tokens = tokenizer(batch_texts)
            tokens = tokens.to(device)

            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            feats = text_features.detach().cpu().numpy().astype("float32")

            for p, vec in zip(batch_paths, feats):
                out_path = embeds_dir / f"{p.stem}.npy"
                np.save(out_path, vec, allow_pickle=False)

    print(f"Done. Wrote {len(files)} embeddings to: {embeds_dir}")

if __name__ == "__main__":
    main()
