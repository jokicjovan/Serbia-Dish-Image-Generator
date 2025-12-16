"""# Imports"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor, Resize
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.nn import functional as F

"""# Config"""

# Device and base model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# Paths for LoRA weights
LORA_ROOT = "/content/drive/MyDrive/Colab Notebooks/sd_lora"
OUTPUT_DIR = f"{LORA_ROOT}/outputs/unet_only"
UNET_LORA_WEIGHTS = f"{OUTPUT_DIR}/checkpoint-step500/unet"
TEXT_ENCODER_LORA_WEIGHTS = f"{OUTPUT_DIR}/checkpoint-step500/text_encoder"

# Dataset paths for testing
TEST_IMAGES_DIR = f"{LORA_ROOT}/dataset/test/images"
TEST_CAPTIONS_DIR = f"{LORA_ROOT}/dataset/test/captions"

# Generation / testing parameters
RESOLUTION = 512
BATCH_SIZE = 2
NUM_INFERENCE_STEPS = 50          # DDIM / DPM solver steps for generation
SAVE_GENERATED_IMAGES = True

# Unified test output directory
TEST_OUTPUT_DIR = f"{LORA_ROOT}/testing/metrics_test/unet_only/checkpoint-step500"
GENERATED_DIR = f"{TEST_OUTPUT_DIR}/generated_images"
Path(GENERATED_DIR).mkdir(parents=True, exist_ok=True)

# Check if LoRA weights for text encoder exist (optional)
USE_TEXT_ENCODER_LORA = Path(TEXT_ENCODER_LORA_WEIGHTS).exists()

"""# Prepare test dataset"""

class ImageCaptionDataset(Dataset):
    def __init__(self, images_dir, captions_dir, resolution=512):
        self.images_dir = Path(images_dir)
        self.captions_dir = Path(captions_dir)
        self.ids = sorted([p.stem for p in self.images_dir.glob("*.jpg")])
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # scale [-1,1]
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img = Image.open(self.images_dir / f"{id_}.jpg").convert("RGB")
        img = self.transform(img)
        caption = (self.captions_dir / f"{id_}.txt").read_text(encoding="utf-8").strip()
        return {"image": img, "caption": caption, "id": id_}

# Load dataset
test_dataset = ImageCaptionDataset(TEST_IMAGES_DIR, TEST_CAPTIONS_DIR, RESOLUTION)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

"""# Load Stable Diffusion + LoRA"""

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
).to(DEVICE)

# Load LoRA weights for UNet
pipe.unet = PeftModel.from_pretrained(pipe.unet, UNET_LORA_WEIGHTS).to(DEVICE)

# Optionally load LoRA weights for Text Encoder
if USE_TEXT_ENCODER_LORA:
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, TEXT_ENCODER_LORA_WEIGHTS).to(DEVICE)
    print("Loaded LoRA weights for Text Encoder.")

# Use DPM solver for faster inference
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Memory optimizations
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()

"""# Initialize metrics"""

# CLIP setup
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# FID metric
fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)

# Transform helpers
to_tensor = ToTensor()
resize = transforms.Resize((RESOLUTION, RESOLUTION))

# Optional: create folder for generated images
if SAVE_GENERATED_IMAGES:
    Path(GENERATED_DIR).mkdir(parents=True, exist_ok=True)

"""# Generate images and compute metrics"""

pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
pipe.enable_attention_slicing()  # reduce VRAM usage

clip_scores = []
clip_cos_sims = []

for batch in tqdm(test_loader, desc="Testing"):
    captions = batch["caption"]
    gt_images = batch["image"].to(DEVICE)  # [-1,1] scale

    # Generate images
    with torch.autocast("cuda"):
        gen_images = pipe(
            captions,
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=NUM_INFERENCE_STEPS
        ).images

    # Convert generated images to tensor [-1,1]
    gen_tensors = torch.stack([to_tensor(resize(img)) * 2 - 1 for img in gen_images]).to(DEVICE)

    # ------------------ CLIPScore & cosine similarity ------------------
    # Preprocess for CLIP
    clip_inputs = clip_processor(
        text=captions,
        images=[Image.fromarray(((img.permute(1,2,0).cpu().numpy()+1)/2*255).astype("uint8")) for img in gen_tensors],
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)
        image_embeds = F.normalize(clip_outputs.image_embeds, dim=-1)
        text_embeds = F.normalize(clip_outputs.text_embeds, dim=-1)

        # Cosine similarity
        cos_sim = (image_embeds * text_embeds).sum(dim=-1)
        clip_cos_sims.extend(cos_sim.cpu().tolist())

        # CLIPScore (scaled 0-100)
        clip_score = ((cos_sim + 1)/2 * 100)
        clip_scores.extend(clip_score.cpu().tolist())

    # ------------------ FID update ------------------
    gt_images_fid = ((gt_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    gen_images_fid = ((gen_tensors + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    fid_metric.update(gt_images_fid, real=True)
    fid_metric.update(gen_images_fid, real=False)

    # ------------------ Save generated images ------------------
    if SAVE_GENERATED_IMAGES:
        for img, id_ in zip(gen_images, batch["id"]):
            img.save(Path(GENERATED_DIR) / f"{id_}_gen.png")

"""# Compute final metrics"""

final_fid = fid_metric.compute().item()
avg_clip_score = sum(clip_scores) / len(clip_scores)
avg_cos_sim = sum(clip_cos_sims) / len(clip_cos_sims)

# Prepare content
metrics_content = f"""
FID score: {final_fid:.4f}
Average CLIPScore (0-100): {avg_clip_score:.2f}
Average CLIP cosine similarity (-1 to 1): {avg_cos_sim:.4f}
"""

# Print metrics
print(metrics_content.strip())

# Save metrics as TXT
metrics_txt_path = Path(GENERATED_DIR).parent / "metrics.txt"
with open(metrics_txt_path, "w") as f:
    f.write(metrics_content.strip())

print(f"Metrics saved to {metrics_txt_path}")