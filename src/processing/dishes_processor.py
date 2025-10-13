import asyncio
import json
import os
import shutil
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# =============== CONFIG =================
BASE_DIR = Path(__file__).resolve().parents[2]

# Input
INPUT_DIR = BASE_DIR / "data/combined"          # Root folder containing raw data (JSON + images)
INPUT_JSON = INPUT_DIR / "dishes.json"          # Input JSON file with dishes

# Output
OUTPUT_DIR = BASE_DIR / "data/processed"        # Root folder for LoRA dataset
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"       # Folder to store copied images
OUTPUT_CAPTIONS_DIR = OUTPUT_DIR / "captions"   # Folder to store generated captions

# OpenAI
MODEL = "gpt-4o-mini"                           # Model for caption generation

# Optional cutoff for number of captions to generate
TARGET_CAPTIONS_COUNT = 1000

# Processing
BATCH_SIZE = 20
CONCURRENT_REQUESTS = 3
RETRY_DELAY = 3
SAVE_EVERY = 5
RESUME_FROM_LAST = True
# ========================================

# Ensure output folders exist
OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============== Helper functions =================

def load_input_dishes():
    """
    Load dishes from JSON and skip already processed ones if RESUME_FROM_LAST.
    """
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
        # print(data)
        # dishes = data.get("dishes", data)
        dishes = data

    print(f"Loaded {len(dishes)} total dishes.")

    remaining_dishes = dishes
    if RESUME_FROM_LAST:
        processed_ids = {f.stem for f in OUTPUT_CAPTIONS_DIR.glob("*.txt")}
        remaining_dishes = [
            d for d in dishes
            if Path(d.get("image_path", "")).stem not in processed_ids
        ]
        skipped = len(dishes) - len(remaining_dishes)
        if skipped:
            print(f"Skipping {skipped} already processed dishes.")

    return remaining_dishes

async def generate_caption(dish, semaphore):
    """
    Generate a caption for a dish using OpenAI and save it as a .txt file.
    Copy the corresponding image to the images folder.
    """
    async with semaphore:
        dish_id = Path(dish.get("image_path", "")).stem
        if not dish_id:
            return None

        prompt_text = f"""
You are a professional culinary caption writer.
The following recipe fields are in Serbian:

Name: {dish.get('name')}
Ingredients: {dish.get('ingredients')}
Preparation: {dish.get('preparation')}

Generate a short English caption (1–2 sentences) suitable for a text-to-image AI model.
Do not include any extra text or JSON. Only output the caption.
"""

        # Retry logic
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.7,
                )
                caption = response.choices[0].message.content.strip()

                # Save caption
                caption_file = OUTPUT_CAPTIONS_DIR / f"{dish_id}.txt"
                caption_file.write_text(caption, encoding="utf-8")

                # Copy image
                src_image = INPUT_DIR / dish.get("image_path", "")
                dst_image = OUTPUT_IMAGES_DIR / f"{dish_id}.jpg"
                if src_image.exists():
                    shutil.copy(src_image, dst_image)
                else:
                    print(f"⚠️ Image not found for {dish['name']}")

                return dish_id

            except Exception as e:
                print(f"⚠️ {dish.get('name')}: Error: {e}, retrying ({attempt + 1}/3)...")
                await asyncio.sleep(RETRY_DELAY)

        print(f"❌ Failed to generate caption for {dish.get('name')}")
        return None

# =============== Main Processing Flow =================

async def main():
    remaining_dishes = load_input_dishes()

    # Optional cutoff
    if TARGET_CAPTIONS_COUNT is not None:
        remaining_dishes = remaining_dishes[:TARGET_CAPTIONS_COUNT]
        print(f"Processing only {len(remaining_dishes)} dishes to reach target of {TARGET_CAPTIONS_COUNT}.")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    if not remaining_dishes:
        print("✅ Nothing to process. All dishes are already processed.")
        return

    completed_batches = 0
    failed_items = 0

    # Create batches
    batches = [remaining_dishes[i:i+BATCH_SIZE] for i in range(0, len(remaining_dishes), BATCH_SIZE)]

    with tqdm(total=len(batches), desc="Processing batches", unit="batch") as pbar:
        for batch_index, batch in enumerate(batches):
            tasks = [generate_caption(d, semaphore) for d in batch]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is None:
                    failed_items += 1

            completed_batches += 1
            pbar.set_postfix({"Success": completed_batches, "Failed": failed_items})
            pbar.update(1)

    print(f"\n✅ Processing finished.")
    print(f"   Completed batches: {completed_batches}")
    print(f"   Failed items: {failed_items}")
    print(f"   Captions saved in: {OUTPUT_CAPTIONS_DIR}")
    print(f"   Images saved in: {OUTPUT_IMAGES_DIR}")

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
