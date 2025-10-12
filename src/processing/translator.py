import asyncio
import json
import os
import re
from openai import AsyncOpenAI
from tqdm import tqdm

# =============== CONFIG =================
INPUT_FILE = "../../data/combined/dishes.json"  # Path to input JSON file
OUTPUT_FILE = "../../data/processed/dishes_processed.json"  # Path to save translated output
MODEL = "gpt-4o-mini"                                 # OpenAI model to use for translation

# TOTAL number of dishes you want translated (including already translated ones).
# For a full translation run, set this number ABOVE the total number of dishes in your dataset.
TARGET_TRANSLATION_COUNT = 100

BATCH_SIZE = 20                                       # Number of dishes per API call
CONCURRENT_REQUESTS = 3                               # Number of parallel translations
RETRY_DELAY = 3                                       # Seconds to wait before retrying a failed batch
SAVE_EVERY = 5                                        # Save progress every N batches
RESUME_FROM_LAST = True                               # Resume from last progress (based on image_path)
# ========================================

# Initialize OpenAI async client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =============== Helper functions =================

def safe_json_load(text: str):
    """
    Safely parse JSON returned from the model.
    Handles cases where the model adds extra commentary outside of JSON.
    """
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def load_input_dishes():
    """
    Loads dishes from the input file.
    If RESUME_FROM_LAST is True and output file exists, it skips already translated ones
    based on unique image_path values.
    """
    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
        dishes = data.get("dishes", data)

    print(f"Loaded {len(dishes)} total dishes.")

    translated_dishes = []
    remaining_dishes = dishes
    if RESUME_FROM_LAST and os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                translated_dishes = existing_data.get("dishes", [])
                print(f"Resuming from previous progress: {len(translated_dishes)} dishes already translated.")

                translated_paths = {
                    d.get("image_path") for d in translated_dishes if d.get("image_path")
                }

                remaining_dishes = [
                    d for d in dishes if d.get("image_path") not in translated_paths
                ]

                skipped = len(dishes) - len(remaining_dishes)
                if skipped:
                    print(f"Skipping {skipped} already translated dishes (based on image_path).")

        except Exception as e:
            print(f"⚠️ Could not read existing translated file ({e}). Starting from scratch.")

    # Limit to target translation count
    translations_needed = TARGET_TRANSLATION_COUNT - len(translated_dishes)
    if translations_needed < len(remaining_dishes):
        remaining_dishes = remaining_dishes[:translations_needed]
        print(f"Translating only {len(remaining_dishes)} dishes to reach target of {TARGET_TRANSLATION_COUNT} translations.")

    return remaining_dishes, translated_dishes


async def translate_batch(batch, semaphore, batch_index):
    """
    Translate a batch of dishes asynchronously.
    Uses semaphore to limit concurrent OpenAI requests.
    """
    async with semaphore:
        # Remove image paths before sending to the model
        batch_without_images = []
        image_paths = []

        for dish in batch:
            dish_copy = dict(dish)  # avoid mutating original
            image_paths.append(dish_copy.pop("image_path", None))
            batch_without_images.append(dish_copy)

        # Prepare translation input
        input_json = json.dumps(batch_without_images, ensure_ascii=False, indent=2)

        # System prompt to ensure clean translation output
        system_prompt = (
            "You are a professional culinary translator. "
            "Translate all Serbian text fields (name, ingredients, preparation) into English. "
            "Keep the JSON structure exactly the same as the input (except image_path which should be ignored). "
            "Return only valid JSON with no extra commentary."
        )

        # Retry logic (up to 3 attempts)
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_json},
                    ],
                    temperature=0.0,
                )

                translated = response.choices[0].message.content.strip()
                translated_batch = safe_json_load(translated)

                # Restore image paths
                for dish, path in zip(translated_batch, image_paths):
                    if path is not None:
                        dish["image_path"] = path

                return batch_index, translated_batch

            except Exception as e:
                print(f"⚠️ Batch {batch_index}: Error: {e}, retrying ({attempt + 1}/3)...")
                await asyncio.sleep(RETRY_DELAY)

        print(f"❌ Batch {batch_index} failed after 3 retries.")
        return batch_index, None


# =============== Main Translation Flow =================

async def main():
    # Load dishes and previous translations (if resuming)
    remaining_dishes, translated_dishes = load_input_dishes()
    already_translated_count = len(translated_dishes)

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []

    # Create translation tasks in batches
    for i in range(0, len(remaining_dishes), BATCH_SIZE):
        batch = remaining_dishes[i : i + BATCH_SIZE]
        batch_index = i // BATCH_SIZE
        tasks.append(translate_batch(batch, semaphore, batch_index))

    completed_batches = 0
    failed_batches = 0

    if not tasks:
        print("✅ Nothing to translate. All dishes are already translated.")
        return

    # Process all batches concurrently with progress bar
    with tqdm(total=len(tasks), desc="Translating in batches", unit="batch") as pbar:
        for coro in asyncio.as_completed(tasks):
            batch_index, result = await coro

            if result is None:
                tqdm.write(f"⚠️ Skipping failed batch {batch_index}")
                failed_batches += 1
            else:
                translated_dishes.extend(result)
                completed_batches += 1

            # Save progress incrementally
            if completed_batches % SAVE_EVERY == 0 and completed_batches > 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
                    json.dump({"dishes": translated_dishes}, output_file, ensure_ascii=False, indent=2)
                tqdm.write(f"💾 Progress saved after {completed_batches} successful batches.")

            # Update live progress bar
            pbar.set_postfix({
                "Success": completed_batches,
                "Failed": failed_batches
            })
            pbar.update(1)

    # Final save of all translated dishes
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        json.dump({"dishes": translated_dishes}, output_file, ensure_ascii=False, indent=2)

    print(f"\n✅ Translation finished.")
    print(f"   Successful batches: {completed_batches}")
    print(f"   Failed batches: {failed_batches}")
    print(f"   Already translated before this run: {already_translated_count}")
    print(f"   Translated in this run: {len(translated_dishes) - already_translated_count}")
    print(f"   Total translated dishes: {len(translated_dishes)}")
    print(f"   Saved to {OUTPUT_FILE}")


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
