import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────
PROCESSED_DIR = "ml/processed"
OUTPUT_DIR    = "ml/processed"

# ── Load data ─────────────────────────────────────────────────────
print("Loading data...")
X = np.load(os.path.join(PROCESSED_DIR, "X_landmarks.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))

with open(os.path.join(PROCESSED_DIR, "label_map.json")) as f:
    label_map = json.load(f)

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
print(f"  Classes : {len(label_map)}")

# ── Split ─────────────────────────────────────────────────────────
# Step 1: split off 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,        # reproducible — same split every time
    stratify=y              # keep class proportions balanced in each split
)

# Step 2: split 30% temp into 15% val + 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# ── Save splits ───────────────────────────────────────────────────
print("\nSaving splits...")
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)

# ── Summary ───────────────────────────────────────────────────────
total = len(X)
print("\n" + "="*50)
print("SPLIT COMPLETE")
print(f"  Total samples : {total:,}")
print(f"  Train         : {len(X_train):,}  ({len(X_train)/total*100:.1f}%)")
print(f"  Validation    : {len(X_val):,}   ({len(X_val)/total*100:.1f}%)")
print(f"  Test          : {len(X_test):,}   ({len(X_test)/total*100:.1f}%)")
print(f"\n  Saved to: {OUTPUT_DIR}/")
print("="*50)

# ── Per-class distribution check ──────────────────────────────────
print("\nPer-class sample count in training set:")
inv_label_map = {v: k for k, v in label_map.items()}
for label_idx in sorted(set(y_train)):
    count = np.sum(y_train == label_idx)
    name  = inv_label_map[label_idx]
    bar   = "█" * (count // 50)
    print(f"  {name:>8} ({label_idx:2d}): {count:4d}  {bar}")