import cv2
import os
import json
import numpy as np

# ------------------------ CONFIG ------------------------
input_dir = "images"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

skipped_file = "skipped.json"
last_file = "last_processed.json"

CROP_SIZE = 512
INITIAL_SCALE = 1.0
SCALE_STEP = 0.1
MOVE_STEP = 20
supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ------------------------ LOAD STATE ------------------------
if os.path.exists(skipped_file):
    with open(skipped_file, "r") as f:
        skipped_images = json.load(f)
else:
    skipped_images = []

if os.path.exists(last_file):
    with open(last_file, "r") as f:
        last_processed = json.load(f)
        last_index = last_processed.get("last_index", -1)
else:
    last_index = -1


# ------------------------ HELPERS ------------------------
def center_crop(img, size):
    h, w = img.shape[:2]
    min_side = min(h, w)
    top = (h - min_side) // 2
    left = (w - min_side) // 2
    cropped = img[top:top + min_side, left:left + min_side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)


def save_skipped(image_name):
    if image_name not in skipped_images:
        skipped_images.append(image_name)
        with open(skipped_file, "w") as f:
            json.dump(skipped_images, f, indent=2)


def update_last_processed(index, filename):
    with open(last_file, "w") as f:
        json.dump({"last_index": index, "last_image": filename}, f, indent=2)


# ------------------------ MANUAL CROP ------------------------
def manual_crop(image, filename):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    min_side = min(h, w)
    scale = INITIAL_SCALE

    def get_box():
        side = int(min_side * scale)
        x1 = np.clip(cx - side // 2, 0, w - side)
        y1 = np.clip(cy - side // 2, 0, h - side)
        x2, y2 = x1 + side, y1 + side
        return x1, y1, x2, y2

    def draw_crop(img):
        x1, y1, x2, y2 = get_box()
        preview = img.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Manual crop", preview)

    cv2.namedWindow("Manual crop")
    draw_crop(image)

    while True:
        key = cv2.waitKey(0) & 0xFF

        # Pan
        if key == ord('w'):
            cy = max(0 + int(min_side * scale) // 2, cy - MOVE_STEP)
        elif key == ord('s'):
            cy = min(h - int(min_side * scale) // 2, cy + MOVE_STEP)
        elif key == ord('a'):
            cx = max(0 + int(min_side * scale) // 2, cx - MOVE_STEP)
        elif key == ord('d'):
            cx = min(w - int(min_side * scale) // 2, cx + MOVE_STEP)

        # Scale
        elif key in [ord('+'), ord('=')]:
            scale = np.clip(scale + SCALE_STEP, 0.1, 1.0)
        elif key in [ord('-'), ord('_')]:
            scale = np.clip(scale - SCALE_STEP, 0.1, 1.0)

        # Confirm / cancel
        elif key in [13, 10]:  # Enter
            x1, y1, x2, y2 = get_box()
            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, crop)
            print(f"✅ Saved manually cropped image: {filename}")
            break
        elif key == 27:  # ESC
            print("❌ Manual crop canceled")
            break
        else:
            pass

        # ignore unknown keys
        draw_crop(image)

    cv2.destroyWindow("Manual crop")


# ------------------------ MAIN LOOP ------------------------
def main():
    files = sorted([f for f in os.listdir(input_dir)
                    if f.lower().endswith(supported_exts)])

    # Start from last processed index + 1
    start_idx = last_index + 1
    idx=start_idx
    while idx<len(files):
        fname = files[idx]
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path) or fname in skipped_images:
            continue

        try:
            path = os.path.join(input_dir, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"Could not read image: {fname}")
                continue

            # Auto center crop preview
            h, w = img.shape[:2]
            min_side = min(h, w)
            x1 = (w - min_side) // 2
            y1 = (h - min_side) // 2
            cropped = img[y1:y1 + min_side, x1:x1 + min_side]
            resized = cv2.resize(cropped, (CROP_SIZE, CROP_SIZE))
            cv2.imshow("Preview", resized)

            print(f"Showing ({idx + 1}/{len(files)}): {fname}")
            print("Controls → W: approve | D: manual crop | S: skip | Q: exit")

            key = cv2.waitKey(0) & 0xFF

            if key == ord('w'):
                cv2.imwrite(out_path, resized)
                print(f"✅ Approved: {fname}")
                idx += 1
            elif key == ord('s'):
                save_skipped(fname)
                print(f"⏭️ Skipped: {fname}")
                idx += 1
            elif key == ord('d'):
                cv2.destroyWindow("Preview")
                manual_crop(img, fname)
                idx += 1
            elif key == ord('q'):
                print(f"⚡ Exiting early at image {idx + 1}/{len(files)} ({fname})")
                update_last_processed(idx, fname)
                break
            else:
                # unknown key pressed → do nothing
                pass

            update_last_processed(idx, fname)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error with {fname}: {e}")
            cv2.destroyAllWindows()

    # Save skipped images to JSON on exit
    with open(skipped_file, "w") as f:
        json.dump(skipped_images, f, indent=2)

    print("✅ Session ended. Skipped images saved.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
