import json
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "combined"

# Input JSONs
file1 = RAW_DIR / "recepti" / "dishes.json"
file2 = RAW_DIR / "coolinarika" / "dishes.json"

# Image source folders
images1 = RAW_DIR / "recepti" / "images"
images2 = RAW_DIR / "coolinarika" / "images"   # fixed typo: "mages" → "images"

# Output paths
output_json = INTERIM_DIR / "dishes.json"
output_images = INTERIM_DIR / "images"

# Create output image folder if it doesn’t exist
output_images.mkdir(parents=True, exist_ok=True)

# === Merge JSONs ===
with open(file1, "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(file2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

merged_data = {"dishes": data1["dishes"] + data2["dishes"]}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"✅ JSON merged successfully into {output_json}")

# === Combine images ===
def copy_images(src_folder, dst_folder):
    count = 0
    for file in os.listdir(src_folder):
        src = src_folder / file
        dst = dst_folder / file

        # Avoid overwriting duplicates
        if dst.exists():
            base, ext = os.path.splitext(file)
            i = 1
            while (dst_folder / f"{base}_{i}{ext}").exists():
                i += 1
            dst = dst_folder / f"{base}_{i}{ext}"

        if src.is_file():
            shutil.copy2(src, dst)
            count += 1
    print(f"📸 Copied {count} images from {src_folder.name}")

copy_images(images1, output_images)
copy_images(images2, output_images)

print(f"✅ All images merged into {output_images}")
