import json
import os
import re
import time
from openai import OpenAI
from tqdm import tqdm

# =============== CONFIG =================
INPUT_FILE = "data/merged.json"
OUTPUT_FILE = "data/merged_translated.json"
MODEL = "gpt-4o-mini"
BATCH_SIZE = 20         # number of dishes per API call
MAX_DISHES = 100        # change to test small subset, or set > len(dishes) for all
SLEEP_BETWEEN_CALLS = 1 # seconds between API calls
# ========================================

# Load API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load dishes JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
    dishes = data.get("dishes", data)  # supports both {"dishes": [...]} or just [...]

print(f"Loaded {len(dishes)} dishes.")

# Apply limit for test mode
if MAX_DISHES < len(dishes):
    dishes = dishes[:MAX_DISHES]
    print(f"Translating only first {MAX_DISHES} dishes (test mode).")

translated_dishes = []
start_index = 0

# =============== Helper functions =================

def safe_json_load(text):
    try:
        return json.loads(text)
    except:
        # Try to extract JSON inside text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise

def translate_batch(batch):
    # Strip image_path before sending for translation
    batch_without_images = []
    image_paths = []

    for dish in batch:
        copy_dish = dict(dish)  # make a copy
        image_paths.append(copy_dish.pop("image_path", None))
        batch_without_images.append(copy_dish)

    input_json = json.dumps(batch_without_images, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are a professional culinary translator. "
        "Translate all Serbian text fields (name, ingredients, preparation) into English. "
        "Keep the JSON structure exactly the same as the input (except image_path which should be ignored). "
        "Return only valid JSON with no extra commentary."
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_json},
                ],
                temperature=0.0,
            )
            translated = response.choices[0].message.content.strip()
            translated_batch = safe_json_load(translated)

            # Restore image_path
            for dish, path in zip(translated_batch, image_paths):
                if path is not None:
                    dish["image_path"] = path

            return translated_batch

        except Exception as e:
            print(f"⚠️ Error: {e}, retrying ({attempt+1}/3)...")
            time.sleep(3)

    raise RuntimeError("Failed after 3 retries.")

# =================================================

# =============== Translation loop =================
for i in tqdm(range(0, len(dishes), BATCH_SIZE), desc="Translating"):
    batch = dishes[i : i + BATCH_SIZE]
    translated_batch = translate_batch(batch)
    translated_dishes.extend(translated_batch)
    time.sleep(SLEEP_BETWEEN_CALLS)

# Save output
output = {"dishes": translated_dishes}
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Translation finished. Saved to {OUTPUT_FILE}")
# =================================================
