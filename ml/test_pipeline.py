import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
import time
import requests
import json

# ── Config ────────────────────────────────────────────────────────
# Switch between local and Railway
#API_BASE_URL     = "http://localhost:8000/api/v1"
API_BASE_URL   = "https://your-railway-url.up.railway.app/api/v1"

EMAIL            = "i.foudhaili25132@pi.tn"
PASSWORD         = "50251838"
HAND_MODEL_PATH  = "ml/hand_landmarker.task"
CONFIDENCE_THRESHOLD = 0.7

# ── Step 1: Authenticate ──────────────────────────────────────────
print("Step 1: Authenticating with API...")
response = requests.post(
    f"{API_BASE_URL}/auth/login",
    json={"email": EMAIL, "password": PASSWORD}
)

if response.status_code != 200:
    print(f"Login failed: {response.json()}")
    exit()

token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print(f"Authenticated successfully.")

# ── Step 2: Verify health ─────────────────────────────────────────
print("\nStep 2: Checking API health...")
health = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health")
print(f"Health: {health.json()}")

# ── Step 3: Download MediaPipe model ─────────────────────────────
print("\nStep 3: Setting up MediaPipe...")
if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        HAND_MODEL_PATH
    )

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
print("MediaPipe ready.")

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

# ── Step 4: Open webcam ───────────────────────────────────────────
print("\nStep 4: Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()
print("Webcam open.")

# ── State ─────────────────────────────────────────────────────────
current_letter   = "—"
current_conf     = 0.0
current_top5     = []
api_status       = "Waiting..."
api_latency_ms   = 0
last_api_call    = 0
API_CALL_INTERVAL = 0.15  # call API every 150ms max
prev_time        = 0

# Pipeline stats
total_frames     = 0
frames_with_hand = 0
api_calls        = 0
api_errors       = 0

print("\nPipeline running. Press Q to quit, F to submit feedback on last prediction.")
print("="*60)

# ── Step 5: Main loop ─────────────────────────────────────────────
with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.time() * 1000)

        # MediaPipe detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_detected = result.hand_landmarks and len(result.hand_landmarks) > 0

        if hand_detected:
            frames_with_hand += 1
            hand = result.hand_landmarks[0]
            draw_landmarks(frame, hand)

            # Extract 63 landmarks
            landmarks = []
            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Call API (rate limited)
            now = time.time()
            if now - last_api_call >= API_CALL_INTERVAL:
                last_api_call = now
                api_calls += 1

                try:
                    t0 = time.time()
                    resp = requests.post(
                        f"{API_BASE_URL}/predict/alphabet",
                        json={"landmarks": landmarks},
                        headers=headers,
                        timeout=5
                    )
                    api_latency_ms = int((time.time() - t0) * 1000)

                    if resp.status_code == 200:
                        data = resp.json()
                        if data["confidence"] >= CONFIDENCE_THRESHOLD:
                            current_letter = data["letter"].upper()
                            current_conf   = data["confidence"]
                            current_top5   = data["top5"]
                            api_status     = f"OK ({api_latency_ms}ms)"
                        else:
                            current_letter = "?"
                            current_conf   = data["confidence"]
                            api_status     = f"Low conf ({api_latency_ms}ms)"
                    else:
                        api_errors += 1
                        api_status = f"Error {resp.status_code}"

                except requests.exceptions.Timeout:
                    api_errors += 1
                    api_status = "Timeout"
                except requests.exceptions.ConnectionError:
                    api_errors += 1
                    api_status = "No connection"

        else:
            current_letter = "—"
            current_conf   = 0.0

        # ── UI ────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Dark right panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-270, 0), (w, h), (15, 21, 55), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Letter
        cv2.putText(frame, "Prediction:", (w-250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,130,180), 1)
        cv2.putText(frame, current_letter, (w-250, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0,255,200), 5)

        # Confidence bar
        bar_w = int(230 * current_conf)
        cv2.rectangle(frame, (w-250, 135), (w-20, 150), (40,40,80), -1)
        cv2.rectangle(frame, (w-250, 135), (w-250+bar_w, 150), (0,188,212), -1)
        cv2.putText(frame, f"{current_conf*100:.1f}%",
                    (w-250, 170), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0,188,212), 1)

        # Top 5
        cv2.putText(frame, "Top 5:", (w-250, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,130,180), 1)
        for i, item in enumerate(current_top5[:5]):
            label = item["letter"].upper()
            prob  = item["confidence"]
            color = (0,255,200) if i == 0 else (150,150,150)
            cv2.putText(frame, f"{i+1}. {label}: {prob*100:.1f}%",
                        (w-250, 225 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # API status
        cv2.putText(frame, f"API: {api_status}",
                    (w-250, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0,255,100) if "OK" in api_status else (0,100,255), 1)

        # Pipeline stats
        cv2.putText(frame,
                    f"Calls: {api_calls}  Errors: {api_errors}",
                    (w-250, h-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90,130,180), 1)

        # Hand status
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

        cv2.putText(frame, "SignVerse Pipeline Test", (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,188,212), 1)

        cv2.imshow("SignVerse — Full Pipeline Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f') and current_top5:
            # Submit feedback for the last prediction
            try:
                fb_resp = requests.post(
                    f"{API_BASE_URL}/predict/feedback",
                    json={
                        "model_type":      "alphabet",
                        "predicted_label": current_top5[0]["letter"],
                        "confidence":      current_top5[0]["confidence"],
                    },
                    headers=headers,
                    timeout=5
                )
                if fb_resp.status_code == 201:
                    print(f"Feedback submitted for '{current_top5[0]['letter']}'")
                    api_status = "Feedback sent!"
            except Exception as e:
                print(f"Feedback error: {e}")

# ── Summary ───────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("PIPELINE TEST SUMMARY")
print(f"  Total frames     : {total_frames}")
print(f"  Frames with hand : {frames_with_hand} ({frames_with_hand/max(total_frames,1)*100:.1f}%)")
print(f"  API calls        : {api_calls}")
print(f"  API errors       : {api_errors}")
print(f"  Error rate       : {api_errors/max(api_calls,1)*100:.1f}%")
print(f"  Last latency     : {api_latency_ms}ms")
print("="*60)