"""# Imports"""

import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from pathlib import Path
from PIL import Image
from tqdm import tqdm

"""# Config"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# LoRA paths
LORA_ROOT = "/content/drive/MyDrive/Colab Notebooks/sd_lora"
UNET_LORA_WEIGHTS = f"{LORA_ROOT}/outputs/unet_te/checkpoint-step500/unet"
TEXT_ENCODER_LORA_WEIGHTS = f"{LORA_ROOT}/outputs/unet_te/checkpoint-step500/text_encoder"

# Output paths
OUTPUT_DIR = f"{LORA_ROOT}/testing/comparison_test/unet_te/checkpoint-step500"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Generation settings
PROMPTS = [
    "Sarma, a traditional Serbian dish of cabbage rolls served in a clay pot",
    "Ćevapi served with flatbread and onions"
]
RESOLUTION = 512
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

"""# Load base Stable Diffusion"""

pipe_base = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
).to(DEVICE)

pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_xformers_memory_efficient_attention()
pipe_base.enable_attention_slicing()

"""# Load LoRA-enhanced Stable Diffusion"""

pipe_lora = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
).to(DEVICE)

# Load UNet LoRA
pipe_lora.unet = PeftModel.from_pretrained(pipe_lora.unet, UNET_LORA_WEIGHTS).to(DEVICE)

# Optionally load Text Encoder LoRA if available
if os.path.exists(TEXT_ENCODER_LORA_WEIGHTS):
    pipe_lora.text_encoder = PeftModel.from_pretrained(pipe_lora.text_encoder, TEXT_ENCODER_LORA_WEIGHTS).to(DEVICE)
    print("Loaded LoRA weights for Text Encoder.")

pipe_lora.scheduler = DPMSolverMultistepScheduler.from_config(pipe_lora.scheduler.config)
pipe_lora.enable_xformers_memory_efficient_attention()
pipe_lora.enable_attention_slicing()

"""# Generate and save images"""

for i, prompt in enumerate(tqdm(PROMPTS, desc="Generating images")):
    # Base SD
    image_base = pipe_base(
        prompt,
        height=RESOLUTION,
        width=RESOLUTION,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE
    ).images[0]
    image_base.save(Path(OUTPUT_DIR) / f"prompt_{i}_base.png")

    # LoRA SD
    image_lora = pipe_lora(
        prompt,
        height=RESOLUTION,
        width=RESOLUTION,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE
    ).images[0]
    image_lora.save(Path(OUTPUT_DIR) / f"prompt_{i}_lora.png")

print(f"Images saved to {OUTPUT_DIR}")