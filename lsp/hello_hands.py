import cv2
import mediapipe as mp
import sys

print("Opening camera... (the first time can take a few seconds)")

# On macOS, force the native AVFoundation backend - it's the reliable one.
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("\n[ERROR] Could not open the camera.")
    print("This is a camera PERMISSION problem, NOT your code.")
    print("Fix it like this:")
    print("  1. Open System Settings > Privacy & Security > Camera")
    print("  2. Turn ON the switch for your terminal app (Terminal / iTerm / VS Code)")
    print("  3. FULLY QUIT that terminal app (Cmd+Q), then reopen it")
    print("  4. Run this script again")
    sys.exit(1)

# Warm up: the first few frames are often empty while the camera wakes up.
print("Camera opened. Warming up...")
for _ in range(10):
    cap.read()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cv2.namedWindow('ASL test - press Q to quit', cv2.WINDOW_NORMAL)
print("A window should be open now. Show your hand to the camera!")
print("Click the window, then press Q to quit.")

fail = 0
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        fail += 1
        if fail > 30:
            print("[ERROR] Camera stopped sending frames. Is another app using it?")
            break
        continue
    fail = 0

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('ASL test - press Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed cleanly. Nice work.")
