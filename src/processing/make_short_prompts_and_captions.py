import asyncio
import json
import os
import re
import hashlib
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# =============== CONFIG =================
INPUT_FILE = "../../data/combined/dishes.json"  # Input JSON
OUTPUT_FILE = "data/merged_short_prompts.json"     # Output JSON
CAPTIONS_DIR = "refined_data/captions"             # Where to write <image_id>.txt
MODEL = "gpt-4o-mini"                              # OpenAI model
TARGET_TRANSLATION_COUNT = 100                     # Total dishes in output
BATCH_SIZE = 20                                    # Dishes per API call
CONCURRENT_REQUESTS = 3                            # Parallel requests
RETRY_DELAY = 3                                    # Seconds before retry
SAVE_EVERY = 5                                     # Save JSON progress every N successful batches
RESUME_FROM_LAST = True                            # Resume based on image_path
OVERWRITE_CAPTIONS = True                          # If caption files already exist, overwrite them
# ========================================

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============== Helper functions =================

def safe_json_load(text: str):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise

def sanitize_token(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untitled"

def infer_image_id(dish: Dict[str, Any], used_ids: set) -> str:
    candidate = None

    if dish.get("id"):
        candidate = sanitize_token(str(dish["id"]))
    elif dish.get("image_path"):
        stem = Path(dish["image_path"]).stem
        candidate = sanitize_token(stem)
    elif dish.get("name"):
        candidate = sanitize_token(dish["name"])

    if not candidate:
        raw = json.dumps(dish, ensure_ascii=False, sort_keys=True)
        candidate = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    base = candidate
    k = 1
    while candidate in used_ids:
        candidate = f"{base}-{k}"
        k += 1

    used_ids.add(candidate)
    return candidate

def load_input_dishes() -> Tuple[list, list]:
    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
        dishes = data.get("dishes", data)

    print(f"Loaded {len(dishes)} total dishes.")

    processed = []
    remaining = dishes

    if RESUME_FROM_LAST and os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f).get("dishes", [])
                processed = existing
                done_paths = {d.get("image_path") for d in existing if d.get("image_path")}
                remaining = [d for d in dishes if d.get("image_path") not in done_paths]
                skipped = len(dishes) - len(remaining)
                if skipped:
                    print(f"Skipping {skipped} already processed dishes (based on image_path).")
        except Exception as e:
            print(f"Could not read existing output file ({e}). Starting from scratch.")

    need = TARGET_TRANSLATION_COUNT - len(processed)
    if need < len(remaining):
        remaining = remaining[:need]
        print(f"Processing only {len(remaining)} dishes to reach target of {TARGET_TRANSLATION_COUNT}.")

    return remaining, processed

async def translate_batch(batch: List[Dict[str, Any]], semaphore: asyncio.Semaphore, batch_index: int):
    async with semaphore:
        batch_without_images = []
        image_paths = []
        originals = []

        for dish in batch:
            dish_copy = dict(dish)
            image_paths.append(dish_copy.pop("image_path", None))
            originals.append(dish)
            batch_without_images.append(dish_copy)

        input_json = json.dumps(batch_without_images, ensure_ascii=False, indent=2)

        system_prompt = (
            "You are a professional culinary translator and prompt writer.\n"
            "Input is a JSON array of dish objects with fields like name, ingredients[], preparation[]. "
            "Text may be in Serbian. For each dish, produce a concise ENGLISH short prompt for image generation.\n\n"
            "Rules for each short prompt (one sentence, 10–25 words):\n"
            "• Start with the translated dish name.\n"
            "• Include cooking method if inferable (baked, fried, boiled, grilled).\n"
            "• Include 2–3 key ingredients WITHOUT quantities/units.\n"
            "• Append 'Serbian cuisine'.\n"
            "• Be factual; no photography/style adjectives.\n\n"
            "Return ONLY a JSON array the same length and order as the input, where each item is {\"short_prompt\": \"...\"}.\n"
            "No extra commentary."
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

                content = response.choices[0].message.content.strip()
                prompts_batch = safe_json_load(content)

                merged = []
                for dish, path, pr in zip(originals, image_paths, prompts_batch):
                    d = dict(dish)
                    if path is not None:
                        d["image_path"] = path
                    d["short_prompt"] = pr.get("short_prompt", "").strip()
                    merged.append(d)

                return batch_index, merged

            except Exception as e:
                print(f"⚠️ Batch {batch_index}: Error: {e}, retrying ({attempt + 1}/3)...")
                await asyncio.sleep(RETRY_DELAY)

        print(f"❌ Batch {batch_index} failed after 3 retries.")
        return batch_index, None

def write_captions(dishes: List[Dict[str, Any]]) -> int:
    out_dir = Path(CAPTIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    used_ids = set()
    written = 0

    for d in dishes:
        sp = (d.get("short_prompt") or "").strip()
        if not sp:
            continue

        image_id = infer_image_id(d, used_ids)
        tgt = out_dir / f"{image_id}.txt"
        if tgt.exists() and not OVERWRITE_CAPTIONS:
            continue

        tgt.write_text(sp, encoding="utf-8")
        written += 1

    return written

async def main():
    remaining, processed = load_input_dishes()
    already_count = len(processed)

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]
        batch_index = i // BATCH_SIZE
        tasks.append(translate_batch(batch, semaphore, batch_index))

    completed_batches = 0
    failed_batches = 0

    if not tasks and processed:
        # Even if nothing to do, ensure captions exist from the processed JSON
        written = write_captions(processed)
        print(f"Nothing new to process. Wrote {written} caption files from existing JSON.")
        return
    elif not tasks:
        print("Nothing to process. Exiting.")
        return

    with tqdm(total=len(tasks), desc="Creating short prompts", unit="batch") as pbar:
        for coro in asyncio.as_completed(tasks):
            batch_index, result = await coro

            if result is None:
                tqdm.write(f"Skipping failed batch {batch_index}")
                failed_batches += 1
            else:
                processed.extend(result)
                completed_batches += 1

            # Incremental save of JSON
            if completed_batches % SAVE_EVERY == 0 and completed_batches > 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
                    json.dump({"dishes": processed}, out, ensure_ascii=False, indent=2)
                tqdm.write(f"Progress saved after {completed_batches} successful batches.")

            pbar.set_postfix({"Success": completed_batches, "Failed": failed_batches})
            pbar.update(1)

    # Final save JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump({"dishes": processed}, out, ensure_ascii=False, indent=2)

    written = write_captions(processed)

    print(f"\n✅ Finished.")
    print(f"   Successful batches: {completed_batches}")
    print(f"   Failed batches: {failed_batches}")
    print(f"   Already processed before this run: {already_count}")
    print(f"   Added in this run: {len(processed) - already_count}")
    print(f"   Total dishes: {len(processed)}")
    print(f"   Caption files written: {written}")
    print(f"   JSON saved to: {OUTPUT_FILE}")
    print(f"   Captions dir: {CAPTIONS_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
