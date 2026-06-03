import numpy as np
import json
import os

# ── Config ────────────────────────────────────────────────────────
PROCESSED_DIR  = "ml/processed"
OUTPUT_DIR     = "ml/processed"
AUGMENT_FACTOR = 2      # Generate 2 augmented copies per sample
NOISE_STD      = 0.005  # How much gaussian noise to add (keep small)
MAX_ROTATION   = 15     # Max rotation angle in degrees

np.random.seed(42)

# ── Load training data only ───────────────────────────────────────
# Never augment val or test — they must stay clean and real
print("Loading training data...")
X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

with open(os.path.join(PROCESSED_DIR, "label_map.json")) as f:
    label_map = json.load(f)

print(f"  Original train shape : {X_train.shape}")

# ── Helper: reshape landmarks ─────────────────────────────────────
# Landmarks are stored flat (63,) — reshape to (21, 3) for transforms
def to_points(row):
    return row.reshape(21, 3)   # (21 landmarks, x/y/z)

def to_flat(points):
    return points.flatten()     # back to (63,)

# ── Augmentation functions ────────────────────────────────────────

def add_noise(row):
    """Add small gaussian noise to all coordinates."""
    points = to_points(row.copy())
    noise  = np.random.normal(0, NOISE_STD, points.shape)
    points += noise
    return to_flat(points)


def random_rotation(row):
    """Rotate hand landmarks by a random angle around the Z axis."""
    points = to_points(row.copy())
    angle  = np.radians(np.random.uniform(-MAX_ROTATION, MAX_ROTATION))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Rotate x and y — leave z unchanged
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    points[:, 0] = cos_a * x - sin_a * y
    points[:, 1] = sin_a * x + cos_a * y

    return to_flat(points)


def mirror_horizontal(row):
    """
    Flip hand horizontally — simulates a mirrored/left-hand signer.
    Simply negate the x coordinate (which is normalized 0-1).
    """
    points = to_points(row.copy())
    points[:, 0] = 1.0 - points[:, 0]   # flip x
    return to_flat(points)


def augment_sample(row):
    """Apply a random combination of augmentations to one sample."""
    augmented = row.copy()

    # Always add a little noise
    augmented = add_noise(augmented)

    # 70% chance of rotation
    if np.random.random() < 0.7:
        augmented = random_rotation(augmented)

    # 40% chance of horizontal mirror
    if np.random.random() < 0.4:
        augmented = mirror_horizontal(augmented)

    return augmented

# ── Generate augmented samples ────────────────────────────────────
print(f"\nGenerating {AUGMENT_FACTOR}x augmented copies...")

augmented_X = []
augmented_y = []

for i in range(len(X_train)):
    for _ in range(AUGMENT_FACTOR):
        aug_sample = augment_sample(X_train[i])
        augmented_X.append(aug_sample)
        augmented_y.append(y_train[i])

augmented_X = np.array(augmented_X, dtype=np.float32)
augmented_y = np.array(augmented_y, dtype=np.int32)

# ── Combine original + augmented ─────────────────────────────────
X_train_aug = np.concatenate([X_train, augmented_X], axis=0)
y_train_aug = np.concatenate([y_train, augmented_y], axis=0)

# Shuffle the combined dataset
shuffle_idx = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[shuffle_idx]
y_train_aug = y_train_aug[shuffle_idx]

# ── Save ──────────────────────────────────────────────────────────
print("\nSaving augmented training data...")
np.save(os.path.join(OUTPUT_DIR, "X_train_aug.npy"), X_train_aug)
np.save(os.path.join(OUTPUT_DIR, "y_train_aug.npy"), y_train_aug)

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("AUGMENTATION COMPLETE")
print(f"  Original train samples  : {len(X_train):,}")
print(f"  Augmented copies added  : {len(augmented_X):,}")
print(f"  Total training samples  : {len(X_train_aug):,}")
print(f"  Augmentation factor     : {AUGMENT_FACTOR + 1}x")
print(f"\n  Saved:")
print(f"    X_train_aug.npy — {X_train_aug.shape}")
print(f"    y_train_aug.npy — {y_train_aug.shape}")
print("="*50)

# ── Per-class check ───────────────────────────────────────────────
print("\nPer-class count in augmented training set:")
inv_label_map = {v: k for k, v in label_map.items()}
for label_idx in sorted(set(y_train_aug)):
    count = int(np.sum(y_train_aug == label_idx))
    name  = inv_label_map[label_idx]
    bar   = "█" * (count // 150)
    print(f"  {name:>8} ({label_idx:2d}): {count:5d}  {bar}")