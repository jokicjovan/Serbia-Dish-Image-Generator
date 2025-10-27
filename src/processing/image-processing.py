from PIL import Image
import os

input_dir = "../../data/images"
output_dir = "dataset/cropped"
os.makedirs(output_dir, exist_ok=True)

target_size = 512  # final size (square)
supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(supported_exts):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"[SKIP] {filename} already exists.")
        continue

    try:
        img = Image.open(input_path).convert("RGB")

        # Center crop to square
        width, height = img.size
        min_side = min(width, height)
        left = (width - min_side) / 2
        top = (height - min_side) / 2
        right = (width + min_side) / 2
        bottom = (height + min_side) / 2
        img = img.crop((left, top, right, bottom))

        # Resize to target
        img = img.resize((target_size, target_size), Image.LANCZOS)
        img.save(output_path)

        print(f"[OK] {filename}")

    except Exception as e:
        print(f"[ERROR] Skipping {filename}: {e}")
