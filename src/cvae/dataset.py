import os,glob, numpy as np, torch
from keras.utils import Sequence
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
class CaptionImageSet(Sequence):

    def __init__(self, root='data/processed', img_size=64, batch_size=32,
                 shuffle=True, test_split=0.1, is_test=False):
        super().__init__()
        self.img_dir = os.path.join(root, "images")
        self.emb_dir = os.path.join(root, "embeds")
        self.img_ids =[]
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.test_split = test_split
        self.is_test = is_test
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.emb_dir):
            raise ValueError(f"Embedding directory not found: {self.emb_dir}")

        self.load_img_ids()
        self._split_data()

        if self.shuffle:
            np.random.shuffle(self.img_ids)

        print(f"Loaded {'test' if is_test else 'train'} dataset: {len(self.img_ids)} samples")

    def load_img_ids(self):
        for p in glob.glob(os.path.join(self.img_dir, "*")):
            base, ext = os.path.splitext(os.path.basename(p))
            if ext.lower() in IMG_EXTS and os.path.exists(os.path.join(self.emb_dir, base + ".npy")):
                self.img_ids.append(base)
        self.img_ids.sort()
        print(f"Found {len(self.img_ids)} valid image-embedding pairs")

    def _split_data(self):
        if self.test_split > 0:
            n_test = int(len(self.img_ids) * self.test_split)
            if self.is_test:
                self.img_ids = self.img_ids[:n_test]
            else:
                self.img_ids = self.img_ids[n_test:]

    def __len__(self):
        return int(np.ceil(len(self.img_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.img_ids))
        batch_ids = self.img_ids[batch_start:batch_end]

        images = []
        embeddings = []

        for img_id in batch_ids:
            img = self._load_image(img_id)
            images.append(img)

            emb = self._load_embedding(img_id)
            embeddings.append(emb)

        images = np.array(images, dtype='float32')
        embeddings = np.array(embeddings, dtype='float32')

        return images, embeddings

    def _load_image(self, img_id):
        img_path = None
        for ext in IMG_EXTS:
            potential_path = os.path.join(self.img_dir, img_id + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        if img_path is None:
            raise ValueError(f"Image not found for ID: {img_id}")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image {img_path}: {e}")

        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)

        img = np.array(img, dtype='float32') / 255.0

        return img

    def _load_embedding(self, img_id):
        emb_path = os.path.join(self.emb_dir, img_id + ".npy")

        try:
            emb = np.load(emb_path).astype('float32')
        except Exception as e:
            raise ValueError(f"Failed to load embedding {emb_path}: {e}")

        return emb.flatten() if emb.ndim>1 else emb

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_ids)

    def get_sample(self, idx):
        img_id = self.img_ids[idx]
        img = self._load_image(img_id)
        emb = self._load_embedding(img_id)
        return img, emb, img_id