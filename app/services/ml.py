import json
import numpy as np
from pathlib import Path
from sqlalchemy.orm import Session
from app.models.feedback import PredictionFeedback
from app.schemas.prediction import FeedbackRequest
from app.models.user import User

# Try lightweight runtime first, fall back to full TF
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
MODEL_PATH     = PROJECT_ROOT / "ml_models" / "alphabet_model.tflite"
LABEL_MAP_PATH = PROJECT_ROOT / "ml_models" / "label_map.json"

# ── Load label map ────────────────────────────────────────────────
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}

# ── Load TFLite model once at startup ────────────────────────────
interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

# Cache these — fetching them on every call adds unnecessary overhead
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index    = input_details[0]['index']
output_index   = output_details[0]['index']

print(f"Alphabet model loaded.")
print(f"  Input  : {input_details[0]['shape']}")
print(f"  Output : {output_details[0]['shape']}")
print(f"  Classes: {len(idx_to_label)}")


def predict_alphabet(landmarks: list[float]) -> dict:
    """
    Accept 63 landmark values (21 landmarks × x,y,z).
    Returns predicted letter, confidence, and top 5.
    """
    if len(landmarks) != 63:
        raise ValueError(f"Expected 63 landmarks, got {len(landmarks)}")

    # Prepare input
    sample = np.array([landmarks], dtype=np.float32)  # (1, 63)

    # Run inference — using cached indices for speed
    interpreter.set_tensor(input_index, sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)[0]

    # Softmax
    exp_out = np.exp(output - np.max(output))
    probs   = exp_out / exp_out.sum()

    # Top prediction
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    letter     = idx_to_label[pred_idx]

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


def save_feedback(db: Session, data: FeedbackRequest, user: User) -> PredictionFeedback:
    """Save a wrong prediction for future retraining."""

    if data.model_type not in ("alphabet", "word"):
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="model_type must be 'alphabet' or 'word'"
        )

    feedback = PredictionFeedback(
        user_id         = user.id,
        model_type      = data.model_type,
        predicted_label = data.predicted_label,
        correct_label   = data.correct_label,
        confidence      = data.confidence,
        landmarks       = json.dumps(data.landmarks) if data.landmarks else None,
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return feedback