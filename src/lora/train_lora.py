"""
LoRA Training Script for Stable Diffusion 1.5
Fine-tune SD1.5 on Serbian traditional dishes dataset
"""

import os
import argparse
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class SerbianDishDataset(Dataset):
    """Dataset for Serbian dish images with captions"""
    def __init__(self, data_root, tokenizer, size=512):
        self.data_root = Path(data_root)
        self.size = size
        self.tokenizer = tokenizer

        # Get all images
        self.image_dir = self.data_root / "images"
        self.caption_dir = self.data_root / "captions"

        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")) +
                                 list(self.image_dir.glob("*.png")))

        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.LANCZOS)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0

        # Load caption
        caption_path = self.caption_dir / f"{img_path.stem}.txt"
        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        else:
            caption = "Serbian traditional dish"

        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids[0],
            "caption": caption
        }


class LoRALayer(torch.nn.Module):
    """LoRA layer for Linear/Conv2d"""
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_down = torch.nn.Linear(in_dim, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_dim, bias=False)

        torch.nn.init.normal_(self.lora_down.weight, std=1/rank)
        torch.nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * self.scale


def inject_lora_to_unet(unet, rank=4, alpha=1.0, target_modules=["to_q", "to_k", "to_v", "to_out.0"], device="cuda", dtype=torch.float32):
    """Inject LoRA layers into U-Net attention layers"""
    lora_layers = []

    for name, module in unet.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, torch.nn.Linear):
                in_features = module.in_features
                out_features = module.out_features

                # Create LoRA layer and move to device with correct dtype
                lora = LoRALayer(in_features, out_features, rank, alpha).to(device=device, dtype=dtype)

                # Store original forward
                original_forward = module.forward

                # Create new forward with LoRA
                def new_forward(x, lora_layer=lora, orig_forward=original_forward):
                    return orig_forward(x) + lora_layer(x)

                module.forward = new_forward
                lora_layers.append((name, lora))

                # Freeze original layer
                module.requires_grad_(False)

    return lora_layers


def train_lora(args):
    import numpy as np

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Stable Diffusion 1.5
    print("Loading Stable Diffusion 1.5...")
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )

    unet = pipe.unet.to(device)
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae.eval()
    text_encoder.eval()

    # Inject LoRA
    print(f"Injecting LoRA with rank={args.rank}, alpha={args.alpha}...")
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    lora_layers = inject_lora_to_unet(unet, rank=args.rank, alpha=args.alpha, device=device, dtype=model_dtype)

    # Collect LoRA parameters
    lora_params = []
    for name, lora in lora_layers:
        lora_params.extend(list(lora.parameters()))

    total_params = sum(p.numel() for p in lora_params)
    print(f"LoRA trainable parameters: {total_params:,}")

    # Dataset
    print("Loading dataset...")
    dataset = SerbianDishDataset(args.data_root, tokenizer, size=args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)

    unet.train()
    global_step = 0

    print("Starting training...")
    progress_bar = tqdm(total=args.max_train_steps, desc="Training")

    while global_step < args.max_train_steps:
        for batch in dataloader:
            # Move to device
            pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
            input_ids = batch["input_ids"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device
            ).long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calculate loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Save checkpoint
            if global_step > 0 and global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"lora_weights_step_{global_step}.pt")
                lora_state = {name: lora.state_dict() for name, lora in lora_layers}
                torch.save(lora_state, save_path)
                print(f"\nSaved checkpoint to {save_path}")

            global_step += 1
            if global_step >= args.max_train_steps:
                break

    # Save final weights
    final_path = os.path.join(args.output_dir, "lora_weights_final.pt")
    lora_state = {name: lora.state_dict() for name, lora in lora_layers}
    torch.save(lora_state, final_path)

    # Save config
    config = {
        "rank": args.rank,
        "alpha": args.alpha,
        "base_model": "runwayml/stable-diffusion-v1-5",
        "resolution": args.resolution,
        "max_train_steps": args.max_train_steps,
    }
    with open(os.path.join(args.output_dir, "lora_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    progress_bar.close()
    print(f"\nTraining complete! Weights saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA for SD1.5")

    # Data
    parser.add_argument("--data_root", type=str, default="data/processed",
                       help="Path to dataset (should contain images/ and captions/ folders)")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Image resolution")

    # LoRA config
    parser.add_argument("--rank", type=int, default=4,
                       help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32.0,
                       help="LoRA alpha")

    # Training
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                       help="Weight decay")
    parser.add_argument("--max_train_steps", type=int, default=1000,
                       help="Max training steps")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Dataloader workers")

    # Saving
    parser.add_argument("--output_dir", type=str, default="runs/lora_sd15",
                       help="Output directory")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")

    args = parser.parse_args()
    train_lora(args)
