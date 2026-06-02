import os
import cv2
import numpy as np
import mediapipe as mp
import urllib.request
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────
DATASET_DIR = "ml/datasets/asl-alphabet/asl_alphabet_train"
OUTPUT_DIR  = "ml/processed"
MODEL_PATH  = "ml/hand_landmarker.task"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Download model if not present ────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded.")

# ── MediaPipe setup ───────────────────────────────────────────────
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,  # Lower threshold for static images
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

# ── Label map ─────────────────────────────────────────────────────
# Sort classes so label indices are always consistent
CLASSES = sorted(os.listdir(DATASET_DIR))
label_map = {cls: idx for idx, cls in enumerate(CLASSES)}

print(f"Classes found: {CLASSES}")
print(f"Label map: {label_map}")

# ── Extraction ────────────────────────────────────────────────────
all_landmarks = []  # Each row = 63 floats (21 landmarks × x,y,z)
all_labels    = []  # Each row = integer class index
skipped       = 0
processed     = 0

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        label = label_map[class_name]

        print(f"\nProcessing class '{class_name}' "
              f"({len(image_files)} images, label={label})")

        for img_file in tqdm(image_files, desc=class_name, unit="img"):
            img_path = os.path.join(class_dir, img_file)

            # Read image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                skipped += 1
                continue

            # Convert BGR → RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Run MediaPipe
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=img_rgb
            )
            result = landmarker.detect(mp_image)

            # Skip if no hand detected
            if not result.hand_landmarks:
                skipped += 1
                continue

            # Extract first hand's 21 landmarks → flatten to 63 floats
            landmarks = result.hand_landmarks[0]
            row = []
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z])  # x, y, z per landmark

            all_landmarks.append(row)
            all_labels.append(label)
            processed += 1

# ── Save as NumPy arrays ──────────────────────────────────────────
X = np.array(all_landmarks, dtype=np.float32)  # Shape: (N, 63)
y = np.array(all_labels,    dtype=np.int32)     # Shape: (N,)

np.save(os.path.join(OUTPUT_DIR, "X_landmarks.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"),    y)

# Save label map so you always know which index = which letter
import json
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"DONE")
print(f"  Processed : {processed:,} images")
print(f"  Skipped   : {skipped:,} images (no hand detected)")
print(f"  X shape   : {X.shape}  (samples × 63 landmarks)")
print(f"  y shape   : {y.shape}")
print(f"  Saved to  : {OUTPUT_DIR}/")
print("="*50)