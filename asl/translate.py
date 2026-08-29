import cv2
import mediapipe as mp
import joblib
import numpy as np
from landmark_utils import (extract_shape, fingertip_xy, motion_features,
                            new_motion_buffer)

MODEL_PATH = "model.pkl"
clf = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("[ERROR] Camera won't open - Settings > Privacy & Security > Camera,")
    print("turn on your terminal app, Cmd+Q it, reopen, try again.")
    raise SystemExit(1)
for _ in range(10):
    cap.read()

cv2.namedWindow("ASL translator - press Q to quit", cv2.WINDOW_NORMAL)
print("Show a sign. For J and Z, perform the motion. Press Q to quit.")

buf = new_motion_buffer()

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    letter = "?"
    conf = 0.0
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        buf.append(fingertip_xy(hand))

        feats = np.array([extract_shape(hand) + motion_features(buf)],
                         dtype=np.float32)
        probs = clf.predict_proba(feats)[0]
        idx = int(np.argmax(probs))
        letter = clf.classes_[idx].upper()
        conf = probs[idx]
    else:
        buf.clear()

    cv2.putText(frame, letter, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
    cv2.putText(frame, f"{conf*100:3.0f}%", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("ASL translator - press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
