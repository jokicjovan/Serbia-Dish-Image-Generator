import os, glob, numpy as np, torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

SUPPORTED_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


class CaptionImageSet(Dataset):
    """Dataset class created to load images and its embeddings"""

    def __init__(self, root='data/processed', img_size=64, test_split=0.1, is_test=False):
        super(CaptionImageSet,self).__init__()
        self.img_dir = os.path.join(root, "images")
        self.emb_dir = os.path.join(root, "embeds")
        self.img_size = img_size
        self.ids = []
        self.test_split = test_split
        self.is_test = is_test
        if is_test:
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.emb_dir):
            raise ValueError(f"Embedding directory not found: {self.emb_dir}")

        self._load_ids()
        self._split_data()

        print(f"Loaded {'test' if is_test else 'train'} dataset: {len(self.ids)} samples")

    def _load_ids(self):
        for file_name in glob.glob(os.path.join(self.img_dir, "*")):
            base_name, extension_name = os.path.splitext(os.path.basename(file_name))
            if extension_name.lower() in SUPPORTED_IMG_EXTENSIONS and os.path.exists(
                    os.path.join(self.emb_dir, base_name + ".npy")):
                self.ids.append(base_name)
        self.ids.sort()
        print(f"Loaded {len(self.ids)} valid pairs of images and embeddings")

    def _split_data(self):
        if self.test_split > 0:
            n_test = int(len(self.ids) * self.test_split)
            if self.is_test:
                self.ids = self.ids[:n_test]
            else:
                self.ids = self.ids[n_test:]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        img_path = None
        img_base_path = os.path.join(self.img_dir, id)
        for ext in SUPPORTED_IMG_EXTENSIONS:
            if os.path.exists(img_base_path + ext):
                img_path = img_base_path + ext
                break

        if img_path is None:
            raise ValueError(f"Image path {id} not found")

        img = Image.open(img_path).convert('RGB')

        embedding = np.load(os.path.join(self.emb_dir, id + ".npy")).astype(np.float32)
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        return self.preprocess(img), torch.from_numpy(embedding)

