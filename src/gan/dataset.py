import os, glob, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

class CaptionImageSet(Dataset):
    def __init__(self, root="data/processed", size=128):
        img_dir = os.path.join(root, "images")
        emb_dir = os.path.join(root, "embedds")

        ids = []
        for p in glob.glob(os.path.join(img_dir, "*")):
            base, ext = os.path.splitext(os.path.basename(p))
            if ext.lower() in IMG_EXTS and os.path.exists(os.path.join(emb_dir, base + ".npy")):
                ids.append(base)
        ids.sort()

        if not ids:
            raise RuntimeError("No image/embedding pairs found. Check data/processed/images and embedds.")

        self.ids = ids
        self.img_dir, self.emb_dir = img_dir, emb_dir
        self.tf = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)  # [-1,1]
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        img = Image.open(os.path.join(self.img_dir, id_ + ".jpg")).convert("RGB") \
              if os.path.exists(os.path.join(self.img_dir, id_ + ".jpg")) \
              else Image.open(next(p for p in glob.glob(os.path.join(self.img_dir, id_ + ".*")) if os.path.splitext(p)[1].lower() in IMG_EXTS)).convert("RGB")

        x = self.tf(img)                          # [3,H,W], in [-1,1]
        e = np.load(os.path.join(self.emb_dir, id_ + ".npy")).astype("float32")
        e /= (np.linalg.norm(e) + 1e-8)
        e = torch.from_numpy(e)                   # [d_clip]
        return x, e
