import os, glob, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

class CaptionImageSet(Dataset):
    def __init__(self, root="data/processed", size=128, embeddings_dir="embedds"):
        img_dir = os.path.join(root, "images")
        # Allow custom embeddings directory name
        emb_dir = os.path.join(root, embeddings_dir)

        ids = []
        for p in glob.glob(os.path.join(img_dir, "*")):
            base, ext = os.path.splitext(os.path.basename(p))
            if ext.lower() in IMG_EXTS and os.path.exists(os.path.join(emb_dir, base + ".npy")):
                ids.append(base)
        ids.sort()

        if not ids:
            raise RuntimeError(f"No image/embedding pairs found. Check {img_dir} and {emb_dir}.")

        self.ids = ids
        self.img_dir, self.emb_dir = img_dir, emb_dir

        if ids:
            first_emb_path = os.path.join(emb_dir, ids[0] + ".npy")
            try:
                first_emb = np.load(first_emb_path)
                print(f"Dataset info: {len(ids)} samples, embedding dim: {first_emb.shape}")
            except:
                print(f"Dataset info: {len(ids)} samples")

        self.tf = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)  # [-1,1]
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]

        # Improved image loading with proper error handling
        try:
            img_path = os.path.join(self.img_dir, id_ + ".jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
            else:
                # Look for other supported formats
                alt_paths = [os.path.join(self.img_dir, id_ + ext) for ext in IMG_EXTS]
                found_path = next((p for p in alt_paths if os.path.exists(p)), None)

                if found_path is None:
                    raise FileNotFoundError(f"No image found for ID '{id_}' in {self.img_dir}")

                img = Image.open(found_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image for ID '{id_}': {str(e)}")

        # Improved embedding loading with error handling
        try:
            emb_path = os.path.join(self.emb_dir, id_ + ".npy")
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"No embedding found for ID '{id_}' at {emb_path}")

            e = np.load(emb_path).astype("float32")

            if e.size == 0:
                raise ValueError(f"Empty embedding file for ID '{id_}'")

            # Normalize embedding
            norm = np.linalg.norm(e)
            if norm < 1e-8:
                raise ValueError(f"Zero or near-zero embedding norm for ID '{id_}'")

            e /= norm
            e = torch.from_numpy(e)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding for ID '{id_}': {str(e)}")

        # Apply image transform
        try:
            x = self.tf(img)
        except Exception as e:
            raise RuntimeError(f"Failed to transform image for ID '{id_}': {str(e)}")

        return x, e
