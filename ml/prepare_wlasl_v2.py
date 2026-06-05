import numpy as np
import json
import os
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────
BASE_DIR       = "ml/datasets/mutemotion-output"
OUTPUT_DIR     = "ml/processed/wlasl_v2"
SEQUENCE_LEN   = 60    # Fixed length — pad/truncate to this
NUM_LANDMARKS  = 180
NUM_COORDS     = 3
FEATURE_SIZE   = NUM_LANDMARKS * NUM_COORDS  # 540 per frame

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load JSON metadata ────────────────────────────────────────────
print("Loading metadata...")
with open(os.path.join(BASE_DIR, "WLASL_parsed_data.json")) as f:
    metadata = json.load(f)

# ── Build label map ───────────────────────────────────────────────
glosses = sorted(set(entry["gloss"] for entry in metadata))
label_map = {gloss: idx for idx, gloss in enumerate(glosses)}
print(f"Total glosses: {len(label_map)}")

# Save label map
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# ── Load landmarks ────────────────────────────────────────────────
print("Loading landmarks_V2.npz (this may take a moment)...")
landmarks_data = np.load(
    os.path.join(BASE_DIR, "landmarks_V2.npz"),
    allow_pickle=True
)
landmark_keys = list(landmarks_data.keys())
print(f"Total landmark arrays: {len(landmark_keys)}")

# ── Helper: pad or truncate to fixed length ───────────────────────
def normalize_sequence(seq, target_len=SEQUENCE_LEN):
    """
    seq shape: (f, 180, 3) — variable f
    Returns: (target_len, 540) — fixed length, flattened landmarks
    """
    # Flatten landmarks: (f, 180, 3) → (f, 540)
    f = seq.shape[0]
    flat = seq.reshape(f, -1).astype(np.float32)

    if f >= target_len:
        # Truncate — evenly sample target_len frames
        indices = np.linspace(0, f - 1, target_len, dtype=int)
        return flat[indices]
    else:
        # Pad with zeros at the end
        pad = np.zeros((target_len - f, FEATURE_SIZE), dtype=np.float32)
        return np.concatenate([flat, pad], axis=0)

# ── Process sequences ─────────────────────────────────────────────
splits = {"train": [], "val": [], "test": []}

print("Processing sequences...")
for i, entry in enumerate(tqdm(metadata)):
    gloss = entry["gloss"]
    split = entry["split"]
    label = label_map[gloss]

    # Get corresponding landmark array by index
    key = landmark_keys[i]
    seq = landmarks_data[key]  # (f, 180, 3)

    if seq is None or len(seq) == 0:
        continue

    normalized = normalize_sequence(seq)  # (60, 540)
    splits[split].append((normalized, label))

# ── Save splits ───────────────────────────────────────────────────
print("\nSaving splits...")
for split_name, items in splits.items():
    if not items:
        continue

    X = np.array([item[0] for item in items], dtype=np.float32)
    y = np.array([item[1] for item in items], dtype=np.int32)

    np.save(os.path.join(OUTPUT_DIR, f"X_{split_name}.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"y_{split_name}.npy"), y)
    print(f"  {split_name}: X={X.shape}  y={y.shape}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("WLASL V2 PREPARATION COMPLETE")
print(f"  Sequence length : {SEQUENCE_LEN} frames")
print(f"  Feature size    : {FEATURE_SIZE} values per frame")
print(f"  Classes         : {len(label_map)}")
print(f"  Saved to        : {OUTPUT_DIR}/")
print("="*50)