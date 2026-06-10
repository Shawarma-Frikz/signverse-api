import json
import numpy as np
from pathlib import Path

# Try lightweight runtime first, fall back to full TF
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "alphabet_model.tflite"
LABEL_MAP_PATH = PROJECT_ROOT / "ml" / "processed" / "label_map.json"

# ── Load label map ────────────────────────────────────────────────
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}

# ── Load TFLite model ─────────────────────────────────────────────
interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Alphabet model loaded.")
print(f"  Input  : {input_details[0]['shape']}")
print(f"  Output : {output_details[0]['shape']}")
print(f"  Classes: {len(idx_to_label)}")


def predict_alphabet(landmarks: list[float]) -> dict:
    """
    Accept 63 landmark values (21 landmarks × x,y,z).
    Returns predicted letter and confidence.
    """
    if len(landmarks) != 63:
        raise ValueError(f"Expected 63 landmarks, got {len(landmarks)}")

    # Prepare input
    sample = np.array([landmarks], dtype=np.float32)  # (1, 63)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Softmax
    exp_out = np.exp(output - np.max(output))
    probs   = exp_out / exp_out.sum()

    # Top prediction
    pred_idx    = int(np.argmax(probs))
    confidence  = float(probs[pred_idx])
    letter      = idx_to_label[pred_idx]

    # Top 5
    top5_indices = np.argsort(probs)[::-1][:5]
    top5 = [
        {"letter": idx_to_label[i], "confidence": float(probs[i])}
        for i in top5_indices
    ]

    return {
        "letter":     letter,
        "confidence": confidence,
        "top5":       top5,
    }