import cv2
import mediapipe as mp
import csv
import os
from landmark_utils import (extract_shape, fingertip_xy, motion_features,
                            new_motion_buffer, feature_names, open_camera)

CSV_PATH = "lsp_data.csv"     # Lengua de Señas Peruana
SAMPLES_PER_PRESS = 100   # snapshots grabbed each time you press a key

# Which keyboard key records which label.
#   a-z  -> letters        0-9 -> numbers
#   ;    -> Ñ  (the ñ key itself does not come through cv2.waitKey reliably,
#               so we use the key that sits in the same place on a US layout)
#   spacebar -> SPACE (thumbs-up)   Tab -> MODE (I-love-you)   Delete -> BACK (thumbs-down)
def key_to_label(key):
    if ord('a') <= key <= ord('z'):
        return chr(key)
    if ord('0') <= key <= ord('9'):
        return chr(key)
    if key in (ord(';'), 241):   # ';' or the latin-1 code for 'ñ'
        return "ñ"
    if key == 32:
        return "SPACE"
    if key == 9:
        return "MODE"
    if key in (8, 127):
        return "BACK"
    return None

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = open_camera()
if not cap.isOpened():
    print("[ERROR] Camera won't open - same permission fix as before.")
    raise SystemExit(1)
for _ in range(10):
    cap.read()

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["label"] + feature_names())

counts = {}
with open(CSV_PATH) as f:
    r = csv.reader(f)
    next(r, None)
    for row in r:
        if row:
            counts[row[0]] = counts.get(row[0], 0) + 1

cv2.namedWindow("Grabar datos LSP - ESC para salir", cv2.WINDOW_NORMAL)
print("TECLAS:  a-z = letras   0-9 = números   ; = Ñ")
print("         ESPACIO = pulgar arriba (espacio)   TAB = cambio de modo   SUPR = borrar")
print("For motion signs (J, Z): press the key, then PERFORM the motion a few times while it records.")
print("Press ESC when done.")

buf = new_motion_buffer()
recording_label = None
remaining = 0

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand = None
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        buf.append(fingertip_xy(hand))
    else:
        buf.clear()

    if recording_label and remaining > 0 and hand is not None:
        feats = extract_shape(hand) + motion_features(buf)
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([recording_label] + feats)
        counts[recording_label] = counts.get(recording_label, 0) + 1
        remaining -= 1
        if remaining == 0:
            print(f"  saved {SAMPLES_PER_PRESS} samples for '{recording_label}' "
                  f"(total {counts[recording_label]})")
            recording_label = None

    if recording_label:
        msg = f"RECORDING '{recording_label}'  {SAMPLES_PER_PRESS - remaining}/{SAMPLES_PER_PRESS}"
        color = (0, 0, 255)
    else:
        msg = "a-z letras | ; = N-tilde | 0-9 numeros | ESPACIO/TAB/SUPR | ESC = salir"
        color = (0, 200, 0)
    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Collected {len(counts)} labels:", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    cv2.putText(frame, " ".join(sorted(counts.keys())), (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("Grabar datos LSP - ESC para salir", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if not recording_label:
        label = key_to_label(key)
        if label is not None:
            recording_label = label
            remaining = SAMPLES_PER_PRESS

cap.release()
cv2.destroyAllWindows()
print("\nSaved to", CSV_PATH)
print("Per-label counts:", {k: counts[k] for k in sorted(counts)})
