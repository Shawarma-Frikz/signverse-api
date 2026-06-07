import cv2
import numpy as np
import mediapipe as mp
import time
import urllib.request
import os
import sys
import torch

# ── Setup ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from tgcn_model import GCN_muti_att

MODEL_PATH      = "ml/models/tgcn_asl2000.pth"
HAND_MODEL_PATH = "ml/hand_landmarker.task"
NUM_SAMPLES     = 50
BUFFER_SIZE     = 60

# ── Load TGCN ─────────────────────────────────────────────────────
model = GCN_muti_att(
    input_feature=100,
    hidden_feature=256,
    num_class=2000,
    p_dropout=0.3,
    num_stage=24
)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()
print("TGCN model loaded.")

# ── Download MediaPipe model if needed ────────────────────────────
if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        HAND_MODEL_PATH
    )

# ── MediaPipe setup ───────────────────────────────────────────────
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
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
        color = (0, 255, 220) if i in (4,8,12,16,20) else (255,255,255)
        cv2.circle(frame, (px, py), 6 if i in (4,8,12,16,20) else 4, color, -1)

def extract_keypoints(frame_bgr, landmarker):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    keypoints = np.zeros((55, 2), dtype=np.float32)
    if result.hand_landmarks:
        for hand_idx, hand in enumerate(result.hand_landmarks[:2]):
            offset = hand_idx * 21
            for lm_idx, lm in enumerate(hand):
                keypoints[offset + lm_idx] = [lm.x, lm.y]
    return keypoints, result.hand_landmarks

def predict(frame_buffer):
    seq = np.stack(frame_buffer)
    indices = np.linspace(0, len(seq)-1, NUM_SAMPLES, dtype=int)
    seq = seq[indices]                                   # (50, 55, 2)
    seq = seq.transpose(1, 0, 2)                         # (55, 50, 2)
    seq = seq.reshape(1, 55, NUM_SAMPLES * 2)            # (1, 55, 100)
    x = torch.FloatTensor(seq)
    with torch.no_grad():
        output = model(x)
        probs  = torch.softmax(output, dim=1)[0]
        top5_vals, top5_idxs = probs.topk(5)
    return [(idx.item(), val.item()) for idx, val in zip(top5_idxs, top5_vals)]

# ── Webcam ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print(f"Webcam open. Press Q to quit.")

frame_buffer  = []
current_top5  = []
current_idx   = "—"
current_conf  = 0.0
prev_time     = 0
hand_detected = False

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        keypoints, landmarks = extract_keypoints(frame, landmarker)
        hand_detected = landmarks is not None and len(landmarks) > 0

        if hand_detected:
            for hand in landmarks:
                draw_landmarks(frame, hand)
            frame_buffer.append(keypoints)
            if len(frame_buffer) > BUFFER_SIZE:
                frame_buffer.pop(0)

        if len(frame_buffer) >= BUFFER_SIZE:
            current_top5 = predict(frame_buffer)
            current_idx  = str(current_top5[0][0])
            current_conf = current_top5[0][1]

        # ── UI ────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (w-260, 0), (w, h), (15, 21, 55), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Buffer bar
        buf_pct = min(len(frame_buffer) / BUFFER_SIZE, 1.0)
        bar_w   = int(220 * buf_pct)
        cv2.rectangle(frame, (w-240, 20), (w-20, 36), (40,40,80), -1)
        cv2.rectangle(frame, (w-240, 20), (w-240+bar_w, 36),
                      (0,188,212) if buf_pct < 1 else (0,255,100), -1)
        cv2.putText(frame, f"{'Buffering' if buf_pct < 1 else 'Ready'} {int(buf_pct*100)}%",
                    (w-240, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,188,212), 1)

        # Prediction — showing class index for now
        cv2.putText(frame, "Class index:", (w-240, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,130,180), 1)
        cv2.putText(frame, current_idx, (w-240, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,200), 3)
        cv2.putText(frame, f"{current_conf*100:.1f}%", (w-240, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,188,212), 2)

        # Top 5 indices
        cv2.putText(frame, "Top 5:", (w-240, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,130,180), 1)
        for i, (idx, prob) in enumerate(current_top5):
            cv2.putText(frame, f"{i+1}. class {idx}: {prob*100:.1f}%",
                        (w-240, 225 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0,255,200) if i == 0 else (150,150,150), 1)

        # Status
        cv2.putText(frame,
                    "Hand detected" if hand_detected else "No hand",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,200) if hand_detected else (100,100,100), 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,200), 2)

        cv2.putText(frame, "SignVerse — WLASL", (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,188,212), 2)

        cv2.imshow("SignVerse — Word Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Done.")