import cv2
import numpy as np
import mediapipe as mp
import json
import time
import urllib.request
import os

# ── Config ────────────────────────────────────────────────────────
TFLITE_PATH   = "ml/models/alphabet_model.tflite"
LABEL_MAP_PATH = "ml/processed/label_map.json"
MODEL_PATH    = "ml/hand_landmarker.task"
CONFIDENCE_THRESHOLD = 0.7

# ── Load TFLite model ─────────────────────────────────────────────
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

interpreter = Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded.")

# ── Load label map ────────────────────────────────────────────────
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
print(f"Labels loaded: {len(idx_to_label)} classes")

# ── Download MediaPipe model if needed ────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

# ── MediaPipe setup ───────────────────────────────────────────────
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 180), 2)
    for i, (px, py) in enumerate(points):
        color = (0, 255, 220) if i in (4, 8, 12, 16, 20) else (255, 255, 255)
        cv2.circle(frame, (px, py), 6 if i in (4,8,12,16,20) else 4, color, -1)

def predict(landmarks_flat):
    sample = np.array([landmarks_flat], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    # Softmax
    exp_out = np.exp(output - np.max(output))
    probs = exp_out / exp_out.sum()
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    return idx_to_label[pred_idx], confidence, probs

# ── Webcam loop ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam open. Press Q to quit.")
prev_time = 0
current_letter = "?"
current_conf   = 0.0
top3           = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_landmarks(frame, hand)

            # Extract 63 values
            landmarks_flat = []
            for lm in hand:
                landmarks_flat.extend([lm.x, lm.y, lm.z])

            letter, conf, probs = predict(landmarks_flat)

            if conf >= CONFIDENCE_THRESHOLD:
                current_letter = letter.upper()
                current_conf   = conf
                # Top 3 predictions
                top_indices = np.argsort(probs)[::-1][:3]
                top3 = [(idx_to_label[i].upper(), probs[i]) for i in top_indices]
            else:
                current_letter = "?"
                current_conf   = conf
                top3 = []
        else:
            current_letter = "—"
            current_conf   = 0.0
            top3 = []

        # ── UI overlay ────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Dark panel on the right
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-220, 0), (w, h), (15, 21, 55), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Big letter display
        cv2.putText(frame, current_letter, (w-180, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 200), 6)

        # Confidence bar
        bar_w = int(180 * current_conf)
        cv2.rectangle(frame, (w-210, 140), (w-30, 160), (40, 40, 80), -1)
        cv2.rectangle(frame, (w-210, 140), (w-210+bar_w, 160), (0, 188, 212), -1)
        cv2.putText(frame, f"{current_conf*100:.1f}%", (w-210, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 188, 212), 2)

        # Top 3
        cv2.putText(frame, "Top 3:", (w-210, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 130, 180), 1)
        for i, (lbl, prob) in enumerate(top3):
            cv2.putText(frame, f"{lbl}: {prob*100:.1f}%",
                        (w-210, 245 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 200) if i == 0 else (150, 150, 150), 1)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        # Title
        cv2.putText(frame, "SignVerse", (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 188, 212), 2)

        cv2.imshow("SignVerse — ASL Alphabet", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Done.")