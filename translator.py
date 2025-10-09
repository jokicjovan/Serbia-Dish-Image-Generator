import asyncio
import json
import os
import re
from openai import AsyncOpenAI
from tqdm import tqdm

# =============== CONFIG =================
INPUT_FILE = "data/merged.json"              # Path to input JSON file
OUTPUT_FILE = "data/merged_translated.json"  # Path to save translated output
MODEL = "gpt-4o-mini"                        # OpenAI model to use for translation
BATCH_SIZE = 20                              # Number of dishes per API call
MAX_DISHES = 100                             # Limit dishes for testing (< total dishes)
CONCURRENT_REQUESTS = 3                      # Number of parallel translations
RETRY_DELAY = 3                              # Seconds to wait before retrying a failed batch
SAVE_EVERY = 10                              # Save progress every N batches
# ========================================

# Initialize OpenAI async client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load input JSON file
with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
    data = json.load(input_file)
    dishes = data.get("dishes", data)

print(f"Loaded {len(dishes)} dishes.")

# Apply limit for testing mode
if MAX_DISHES < len(dishes):
    dishes = dishes[:MAX_DISHES]
    print(f"Translating only first {MAX_DISHES} dishes (test mode).")


# =============== Helper functions =================

def safe_json_load(text):
    """
    Safely parse JSON returned from model.
    Handles cases where model adds extra commentary outside of JSON.
    """
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


async def translate_batch(batch, semaphore, batch_index):
    """
    Translate a batch of dishes asynchronously, using a semaphore to limit concurrency.

    Returns:
        tuple(batch_index, translated_batch or None)
    """
    async with semaphore:
        batch_without_images = []
        image_paths = []

        # Remove image_path to reduce payload
        for dish in batch:
            dish_copy = dict(dish)  # avoid mutating original
            image_paths.append(dish_copy.pop("image_path", None))
            batch_without_images.append(dish_copy)

        input_json = json.dumps(batch_without_images, ensure_ascii=False, indent=2)

        system_prompt = (
            "You are a professional culinary translator. "
            "Translate all Serbian text fields (name, ingredients, preparation) into English. "
            "Keep the JSON structure exactly the same as the input (except image_path which should be ignored). "
            "Return only valid JSON with no extra commentary."
        )

        # Retry logic for reliability
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
                print(f"⚠️ Batch {batch_index}: Error: {e}, retrying ({attempt+1}/3)...")
                await asyncio.sleep(RETRY_DELAY)

        print(f"❌ Batch {batch_index} failed after 3 retries.")
        return batch_index, None


async def main():
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []

    # Create translation tasks in batches
    for i in range(0, len(dishes), BATCH_SIZE):
        batch = dishes[i : i + BATCH_SIZE]
        batch_index = i // BATCH_SIZE
        tasks.append(translate_batch(batch, semaphore, batch_index))

    translated_dishes = []
    completed_batches = 0
    failed_batches = 0

    # Dynamic progress bar
    with tqdm(total=len(tasks), desc="Translating in batches", unit="batch") as pbar:
        for coro in asyncio.as_completed(tasks):
            batch_index, result = await coro

            if result is None:
                tqdm.write(f"⚠️ Skipping failed batch {batch_index}")
                failed_batches += 1
            else:
                translated_dishes.extend(result)
                completed_batches += 1

            # Save progress every SAVE_EVERY batches (successful + failed)
            total_batches = completed_batches + failed_batches
            if total_batches % SAVE_EVERY == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
                    json.dump({"dishes": translated_dishes}, output_file, ensure_ascii=False, indent=2)
                tqdm.write(f"💾 Progress saved after {total_batches} batches (completed: {completed_batches}, failed: {failed_batches}).")

            # Update progress bar stats
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
    print(f"   Total dishes translated: {len(translated_dishes)}")
    print(f"   Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
