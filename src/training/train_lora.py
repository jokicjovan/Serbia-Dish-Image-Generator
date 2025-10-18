from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from peft import LoraConfig, get_peft_model

# =================== CONFIG ===================
ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "data" / "processed"           # expects images/ and captions/ subfolders
SAVE_DIR = ROOT_DIR / "model" / "lora_model"

PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 512
BATCH_SIZE = 1
EPOCHS = 6
LR = 1e-4
GRADIENT_ACCUMULATION_STEPS = 1     # simulate batch size = BATCH_SIZE * this
SAVE_EVERY_EPOCH = True

SEED = 42
torch.manual_seed(SEED)

# =================== DATASET ===================
class DishDataset(Dataset):
    """
    Expects:
      DATASET_DIR/images/*.jpg
      DATASET_DIR/captions/*.txt
    """
    def __init__(self, dataset_dir, image_size=IMAGE_SIZE):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.captions_dir = self.dataset_dir / "captions"
        self.ids = [p.stem for p in self.images_dir.glob("*.jpg")]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img_path = self.images_dir / f"{id_}.jpg"
        txt_path = self.captions_dir / f"{id_}.txt"

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        return {"image": image, "caption": caption}

# =================== MAIN FUNCTION ===================
def main():
    # ---------------- DataLoader ----------------
    dataset = DishDataset(DATASET_DIR, IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Must be 0 on Windows to avoid spawn errors
        pin_memory=True if DEVICE == "cuda" else False,
        drop_last=True
    )

    # ---------------- Load models ----------------
    print("Loading base model components...")
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL, subfolder="unet").to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL, subfolder="text_encoder").to(DEVICE)

    # ---------------- LoRA adapters ----------------
    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_v"],
        lora_dropout=0.05,
        bias="none"
    )

    text_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    print("Applying LoRA adapters...")
    unet = get_peft_model(unet, unet_lora_config)
    text_encoder = get_peft_model(text_encoder, text_lora_config)

    # Enable gradient checkpointing to save memory
    try:
        unet.enable_gradient_checkpointing()
    except Exception:
        pass
    try:
        text_encoder.gradient_checkpointing_enable()
    except Exception:
        pass

    # ---------------- Scheduler & Optimizer ----------------
    print("Loading scheduler...")
    scheduler = PNDMScheduler.from_pretrained(PRETRAINED_MODEL, subfolder="scheduler")

    trainable_params = (
        [p for p in unet.parameters() if p.requires_grad] +
        [p for p in text_encoder.parameters() if p.requires_grad]
    )
    optimizer = optim.AdamW(trainable_params, lr=LR, betas=(0.9, 0.999), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # ---------------- Training Loop ----------------
    print(f"Starting training on {DEVICE} — epochs: {EPOCHS}, batch_size: {BATCH_SIZE}, grad_accum: {GRADIENT_ACCUMULATION_STEPS}")

    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loop):
            images = batch["image"].to(DEVICE, non_blocking=True)
            captions = batch["caption"]

            # Tokenize captions
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            input_ids = text_inputs["input_ids"].to(DEVICE)
            attention_mask = text_inputs["attention_mask"].to(DEVICE)

            # Encode text
            text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state

            # Encode images to latents (no grad)
            with torch.no_grad():
                vae_out = vae.encode(images).latent_dist
                latents = vae_out.sample() * 0.18215

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                unet_output = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds)
                noise_pred = getattr(unet_output, "sample", unet_output)
                loss = nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # Scale loss for accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS

            if DEVICE == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
                if DEVICE == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            loop.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1} finished — avg loss: {avg_loss:.4f}")

        # Save checkpoint each epoch
        if SAVE_EVERY_EPOCH:
            epoch_dir = SAVE_DIR / f"epoch_{epoch+1}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving adapters to {epoch_dir} ...")
            unet.save_pretrained(epoch_dir / "unet")
            text_encoder.save_pretrained(epoch_dir / "text_encoder")
            tokenizer.save_pretrained(epoch_dir / "tokenizer")

    # Final save
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print("Saving final LoRA adapters...")
    unet.save_pretrained(SAVE_DIR / "unet")
    text_encoder.save_pretrained(SAVE_DIR / "text_encoder")
    tokenizer.save_pretrained(SAVE_DIR / "tokenizer")

    print("✅ Training complete!")

# =================== ENTRY POINT ===================
if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
