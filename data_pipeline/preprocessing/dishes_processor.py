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
INPUT_DIR = BASE_DIR / "data/combined"          # Root directory containing raw data (JSON + images)
INPUT_JSON = INPUT_DIR / "dishes.json"          # Input JSON file with dishes array

# Output
OUTPUT_DIR = BASE_DIR / "data/processed"        # Root directory for processed data
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"       # Directory to store images
OUTPUT_CAPTIONS_DIR = OUTPUT_DIR / "captions"   # Directory to store generated captions

# OpenAI
MODEL = "gpt-4o-mini"                           # Model for caption generation

# TOTAL number of captions to have in the end (including already processed ones)
CAPTIONS_GENERATE_CUTOFF = 10000

# Processing
CONCURRENT_REQUESTS = 3                         # Number of parallel translations
RETRY_DELAY = 3                                 # Seconds to wait before retrying a failed batch
RESUME_FROM_LAST = True                         # Resume from last progress (based on image_path)
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
    print("📂 Loading dishes from JSON...")
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        dishes = json.load(f)

    print(f"   → Loaded {len(dishes)} total dishes from {INPUT_JSON}")

    remaining_dishes = dishes
    if RESUME_FROM_LAST:
        processed_ids = {f.stem for f in OUTPUT_CAPTIONS_DIR.glob("*.txt")}
        remaining_dishes = [
            d for d in dishes
            if Path(d.get("image_path", "")).stem not in processed_ids
        ]
        skipped = len(dishes) - len(remaining_dishes)
        if skipped:
            print(f"   ⚙️  Skipping {skipped} already processed dishes.")
        else:
            print("   ✅ No previously processed dishes found — starting fresh.")

    return remaining_dishes


def apply_cutoff(dishes, cutoff, output_dir):
    """
    Applies CAPTIONS_GENERATE_CUTOFF so that total (processed + new) <= cutoff.
    """
    if cutoff is None:
        return dishes  # No limit

    # Count already processed
    processed_count = len(list(output_dir.glob("*.txt")))
    remaining_quota = max(0, cutoff - processed_count)

    if remaining_quota == 0:
        print(f"✅ Target of {cutoff} already reached. Skipping all.")
        return []

    if len(dishes) > remaining_quota:
        print(f"📊 Limiting to {remaining_quota} dishes for processing to reach {cutoff} total.")
        dishes = dishes[:remaining_quota]
    else:
        print(f"📊 Processing all {len(dishes)} remaining dishes "
              f"(target {cutoff}, already {processed_count}).")

    return dishes


async def generate_caption(dish, semaphore):
    """
    Generate a caption for a dish using OpenAI and save it as a .txt file.
    Copy the corresponding image to the images' folder.
    """
    async with semaphore:
        dish_id = Path(dish.get("image_path", "")).stem
        dish_name = dish.get("name", "Unknown Dish")

        if not dish_id:
            print(f"⚠️ Skipping dish without image_path: {dish_name}")
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
                    print(f"⚠️ Image missing for '{dish_name}' → {src_image}")

                return dish_id

            except Exception as e:
                print(f"⚠️ {dish.get('name')}: Error: {e}, retrying ({attempt + 1}/3)...")
                await asyncio.sleep(RETRY_DELAY)

        print(f"❌ Failed to generate caption for {dish.get('name')}")
        return None


# =============== Main Processing Flow =================

async def main():
    remaining_dishes = load_input_dishes()
    remaining_dishes = apply_cutoff(remaining_dishes, CAPTIONS_GENERATE_CUTOFF, OUTPUT_CAPTIONS_DIR)

    if not remaining_dishes:
        print("✅ Nothing to process. All dishes are already processed or quota reached.")
        return

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    successful = 0
    failed = 0

    with tqdm(total=len(remaining_dishes), desc="Generating captions", unit="dish") as pbar:
        tasks = [generate_caption(d, semaphore) for d in remaining_dishes]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is None:
                failed += 1
            else:
                successful += 1
            pbar.set_postfix({"Success": successful, "Failed": failed})
            pbar.update(1)

    print("\n🏁 Caption generation completed.")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"📁 Captions saved in: {OUTPUT_CAPTIONS_DIR}")
    print(f"🖼️ Images copied to: {OUTPUT_IMAGES_DIR}")


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
