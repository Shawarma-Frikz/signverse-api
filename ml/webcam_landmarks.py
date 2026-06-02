import cv2
import mediapipe as mp
import time
import urllib.request
import os

# ── Download model if not present ────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded.")

# ── Hand connections (21 landmarks, standard MediaPipe layout) ────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # Thumb
    (0,5),(5,6),(6,7),(7,8),         # Index
    (0,9),(9,10),(10,11),(11,12),    # Middle
    (0,13),(13,14),(14,15),(15,16),  # Ring
    (0,17),(17,18),(18,19),(19,20),  # Pinky
    (5,9),(9,13),(13,17),            # Palm
]

# ── MediaPipe setup ───────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Webcam setup ──────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam open. Press Q to quit.")
prev_time = 0


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape

    # Convert normalized coords to pixel coords
    points = [
        (int(lm.x * w), int(lm.y * h))
        for lm in hand_landmarks
    ]

    # Draw connections
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 180), 2)

    # Draw landmark dots
    for i, (px, py) in enumerate(points):
        # Fingertips (4,8,12,16,20) slightly bigger + brighter
        if i in (4, 8, 12, 16, 20):
            cv2.circle(frame, (px, py), 7, (0, 255, 220), -1)
        else:
            cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)


with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        # Draw landmarks
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(frame, hand_landmarks)

                # Print coordinates to terminal
                for idx, lm in enumerate(hand_landmarks):
                    print(f"  Landmark {idx:2d}: "
                          f"x={lm.x:.3f}  y={lm.y:.3f}  z={lm.z:.3f}")

            num_hands = len(result.hand_landmarks)
            label = f"{num_hands} hand{'s' if num_hands > 1 else ''} detected"
            cv2.putText(frame, label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        else:
            cv2.putText(frame, "No hands detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        cv2.imshow("SignVerse Hand Landmark Visualizer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Done.")