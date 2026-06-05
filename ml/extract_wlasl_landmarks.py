import os
import cv2
import numpy as np
import mediapipe as mp
import json
import urllib.request
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────
VIDEOS_DIR        = "ml/datasets/wlasl-processed/videos"
JSON_PATH         = "ml/datasets/wlasl-processed/WLASL_v0.3.json"
ALL_GLOSSES_PATH  = "ml/datasets/wlasl-processed/all_glosses.json"
OUTPUT_DIR        = "ml/processed/wlasl"
MODEL_PATH        = "ml/hand_landmarker.task"
SEQUENCE_LEN      = 30   # Fixed number of frames per sequence

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

# ── Load data ─────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    data = json.load(f)

with open(ALL_GLOSSES_PATH) as f:
    all_glosses = json.load(f)

# Build label map from all 2000 glosses
label_map = {gloss: idx for idx, gloss in enumerate(all_glosses)}
print(f"Total glosses: {len(label_map)}")

# Available videos on disk
available_videos = set(f.split(".")[0] for f in os.listdir(VIDEOS_DIR))
print(f"Videos on disk: {len(available_videos)}")

# ── MediaPipe setup ───────────────────────────────────────────────
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

# ── Helper: extract landmarks from one frame ──────────────────────
def extract_frame_landmarks(frame_bgr, landmarker):
    """
    Returns a flat array of 126 floats (2 hands × 21 landmarks × x,y,z)
    If only one hand detected, second hand is zeros.
    If no hands detected, returns None.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    # Always output 2 hands worth of landmarks (pad with zeros if only 1)
    all_landmarks = np.zeros(126, dtype=np.float32)  # 2 × 21 × 3

    for hand_idx, hand in enumerate(result.hand_landmarks[:2]):
        offset = hand_idx * 63  # 21 landmarks × 3 coords
        for lm_idx, lm in enumerate(hand):
            base = offset + lm_idx * 3
            all_landmarks[base]     = lm.x
            all_landmarks[base + 1] = lm.y
            all_landmarks[base + 2] = lm.z

    return all_landmarks


# ── Helper: process one video ─────────────────────────────────────
def process_video(video_path, frame_start, frame_end, landmarker):
    """
    Extract SEQUENCE_LEN evenly-spaced frames from the video.
    Returns array of shape (SEQUENCE_LEN, 126) or None if failed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine frame range
    start = max(0, frame_start - 1)
    end   = total_frames if frame_end == -1 else min(frame_end, total_frames)

    if end <= start:
        cap.release()
        return None

    # Sample SEQUENCE_LEN evenly-spaced frame indices
    frame_indices = np.linspace(start, end - 1, SEQUENCE_LEN, dtype=int)

    sequence = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            sequence.append(np.zeros(126, dtype=np.float32))
            continue

        landmarks = extract_frame_landmarks(frame, landmarker)
        if landmarks is None:
            sequence.append(np.zeros(126, dtype=np.float32))
        else:
            sequence.append(landmarks)

    cap.release()
    return np.array(sequence, dtype=np.float32)  # (30, 126)


# ── Main extraction loop ──────────────────────────────────────────
all_sequences = []   # Each item: (SEQUENCE_LEN, 126)
all_labels    = []   # Each item: int label
splits        = []   # Each item: "train" / "val" / "test"
skipped       = 0
processed     = 0

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    for entry in tqdm(data, desc="Glosses"):
        gloss = entry["gloss"]

        # Skip if not in our label map
        if gloss not in label_map:
            continue

        label = label_map[gloss]

        for instance in entry["instances"]:
            video_id = instance["video_id"]

            # Skip if video not on disk
            if video_id not in available_videos:
                skipped += 1
                continue

            # Find the actual file (could be .mp4 or other ext)
            video_file = None
            for ext in [".mp4", ".avi", ".mov"]:
                candidate = os.path.join(VIDEOS_DIR, video_id + ext)
                if os.path.exists(candidate):
                    video_file = candidate
                    break

            if video_file is None:
                skipped += 1
                continue

            sequence = process_video(
                video_file,
                instance["frame_start"],
                instance["frame_end"],
                landmarker
            )

            if sequence is None:
                skipped += 1
                continue

            all_sequences.append(sequence)
            all_labels.append(label)
            splits.append(instance["split"])
            processed += 1

# ── Save ──────────────────────────────────────────────────────────
print("\nSaving...")

X = np.array(all_sequences, dtype=np.float32)  # (N, 30, 126)
y = np.array(all_labels,    dtype=np.int32)     # (N,)

# Split by original WLASL split
for split_name in ["train", "val", "test"]:
    idx = [i for i, s in enumerate(splits) if s == split_name]
    np.save(os.path.join(OUTPUT_DIR, f"X_{split_name}.npy"), X[idx])
    np.save(os.path.join(OUTPUT_DIR, f"y_{split_name}.npy"), y[idx])
    print(f"  {split_name}: {len(idx)} sequences")

# Save label map
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("WLASL EXTRACTION COMPLETE")
print(f"  Processed : {processed} sequences")
print(f"  Skipped   : {skipped} videos")
print(f"  X shape   : {X.shape}  (sequences × frames × landmarks)")
print(f"  Saved to  : {OUTPUT_DIR}/")
print("="*50)