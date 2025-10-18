import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from pathlib import Path

# ================= CONFIG =================
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
ROOT_DIR = Path(__file__).resolve().parents[1]
LORA_UNET_PATH = ROOT_DIR / "model/lora_model/unet"
LORA_TEXT_ENCODER_PATH = ROOT_DIR / "model/lora_model/text_encoder"
PROMPT = "Sarma traditional Serbian dish on a wooden plate"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "sarma.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD PIPELINE =================
print("🔄 Loading base pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# ================= APPLY LORA WEIGHTS =================
print("🔗 Applying LoRA weights...")
# Load LoRA adapters for UNet
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_UNET_PATH)
# Load LoRA adapters for text encoder
pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, LORA_TEXT_ENCODER_PATH)

# Enable memory-efficient attention if supported
if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    pipe.enable_xformers_memory_efficient_attention()

# ================= GENERATE IMAGE =================
print(f"🎨 Generating image for prompt: {PROMPT}")
with torch.autocast("cuda", enabled=(DEVICE == "cuda")):
    image = pipe(PROMPT, num_inference_steps=30, guidance_scale=7.5).images[0]

# ================= SAVE IMAGE =================
image.save(OUTPUT_FILE)
print(f"✅ Done! Saved image to: {OUTPUT_FILE}")
