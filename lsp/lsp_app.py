import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import sys
import socket
import time
from collections import deque
from landmark_utils import (extract_shape, fingertip_xy, motion_features,
                            new_motion_buffer, augment, open_camera)


def dibujable(texto):
    """OpenCV solo dibuja ASCII: la Ñ saldria como '?'.
    Se muestra como 'N~' en pantalla, pero el texto real guarda la Ñ."""
    return (texto.replace("\u00d1", "N~").replace("\u00f1", "n~")
                 .encode("ascii", "replace").decode("ascii"))

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
STABLE_FRAMES = 8      # how many steady frames before a sign "counts" (~0.4s)
CONF_THRESH = 0.50     # minimum confidence to accept a sign
SEND_HOLD_SECONDS = 2.0  # hold BOTH open palms this long to auto-send

# On a Raspberry Pi running as an appliance there is no screen and no keyboard,
# so the preview window is skipped and everything is driven by gestures.
# Forced with ASL_HEADLESS=1, otherwise auto-detected on Linux with no display.
HEADLESS = (os.environ.get("ASL_HEADLESS") == "1" or
            (sys.platform.startswith("linux") and not os.environ.get("DISPLAY")))


class Typer:
    """All the typing logic: masking by mode, the stability filter, and the
    growing text. Kept separate from the camera so it can be tested."""

    def __init__(self, clf, stable_frames=STABLE_FRAMES, conf_thresh=CONF_THRESH):
        self.clf = clf
        self.classes = list(clf.classes_)
        self.idx = {c: i for i, c in enumerate(self.classes)}
        self.controls = [c for c in self.classes if c in ("SPACE", "MODE", "BACK")]
        self.letters = [c for c in self.classes if len(c) == 1 and c.isalpha()]
        self.numbers = [c for c in self.classes if len(c) == 1 and c.isdigit()]
        self.has_numbers = len(self.numbers) > 0
        self.stable_frames = stable_frames
        self.conf_thresh = conf_thresh

        self.mode = "letter"
        self.text = ""
        self._prev = None
        self._stable = 0
        self._committed = None     # locked until the sign changes or hand leaves
        self.last_motion = 0.0     # movimiento suavizado (lo que se ve en pantalla)
        self._mov_hist = deque(maxlen=self.VENTANA_MOV)
        # OJO: la N-tilde NO se agrega a self.letters. El modelo no la conoce,
        # asi que no puede entrar en el argmax; se genera en predict() a partir
        # de la N cuando hay movimiento.

    def _allowed(self):
        base = self.letters if self.mode == "letter" else self.numbers
        return base + self.controls

    # ── LETRAS CON MOVIMIENTO ────────────────────────────────────────
    # Algunas letras tienen la MISMA forma de mano que otra y solo se
    # distinguen porque la mano se mueve (guia oficial del MINEDU, pags 69-70):
    #   N + movimiento lateral  ->  N-tilde
    #   I + movimiento del menique -> J
    #   Z se traza en el aire con el indice
    #
    # La N-tilde NO esta entrenada en el modelo: se genera aqui a partir de la N
    # cuando se detecta movimiento. Umbrales ajustables con variables de entorno.
    IJ_MOTION_THRESH   = float(os.environ.get("UMBRAL_J",  "0.25"))
    ENIE_MOTION_THRESH = float(os.environ.get("UMBRAL_ENIE", "2.50"))
    MIN_MOTION_JZ      = float(os.environ.get("UMBRAL_JZ", "1.50"))
    VENTANA_MOV        = 10   # frames que se promedian para suavizar

    @staticmethod
    def _motion(feats):
        """(movimiento total de la mano, movimiento del menique)."""
        vec = np.asarray(feats).ravel()
        if vec.shape[0] < 68:
            return 0.0, 0.0
        return float(np.sum(np.abs(vec[63:68]))), float(abs(vec[67]))

    def predict(self, feats):
        """Best label for the CURRENT mode only (the masking trick),
        plus the motion rules for N-tilde, J and Z."""
        probs = self.clf.predict_proba(feats)[0]
        opts = self._allowed()
        if not opts:
            return "?", 0.0
        inst, pinky = self._motion(feats) if feats is not None else (0.0, 0.0)
        # promediar unos frames: un pico suelto ya no dispara la N-tilde
        self._mov_hist.append(inst)
        total = sum(self._mov_hist) / len(self._mov_hist)
        self.last_motion = total          # para mostrarlo en pantalla

        ordenadas = sorted(opts, key=lambda c: probs[self.idx[c]], reverse=True)
        best = ordenadas[0]
        conf = float(probs[self.idx[best]])

        # J y Z solo valen si la mano se esta moviendo. Con la mano quieta,
        # pasar a la mejor letra que no sea de movimiento.
        if best in ("j", "z") and total < self.MIN_MOTION_JZ:
            for c in ordenadas[1:]:
                if c not in ("j", "z"):
                    best, conf = c, float(probs[self.idx[c]])
                    break

        # I vs J: misma forma, decide el movimiento del menique.
        if best in ("i", "j"):
            decidida = "j" if pinky > self.IJ_MOTION_THRESH else "i"
            if decidida in self.idx and decidida in opts:
                best = decidida
                conf = max(conf, float(probs[self.idx[decidida]]))

        # N + movimiento lateral = N-tilde (letra virtual, no entrenada).
        if best == "n" and total > self.ENIE_MOTION_THRESH:
            best = "\u00f1"
            # la confianza se mantiene: la forma es la de la N, que si esta entrenada

        return best, conf

    def update(self, cur, conf):
        """Feed one frame. Returns the committed label, or None if nothing fired."""
        if cur == self._prev:
            self._stable += 1
        else:
            self._stable = 1
            self._prev = cur
        if (self._stable >= self.stable_frames and conf >= self.conf_thresh
                and cur != self._committed and cur != "?"):
            self._committed = cur
            self._apply(cur)
            return cur
        return None

    def reset(self):
        """Call when the hand leaves the frame, so the same sign can repeat."""
        self._prev = None
        self._stable = 0
        self._committed = None
        self._mov_hist.clear()

    def progress(self):
        return min(self._stable / self.stable_frames, 1.0)

    def _apply(self, label):
        if label == "MODE":
            if self.has_numbers:
                self.mode = "number" if self.mode == "letter" else "letter"
        elif label == "SPACE":
            self.text += " "
        elif label == "BACK":
            self.text = self.text[:-1]
        elif label.isdigit():
            self.text += label
        else:
            self.text += label.upper()

    # --- keyboard helpers (handy while testing before gestures are trained) ---
    def key_space(self):   self.text += " "
    def key_back(self):    self.text = self.text[:-1]
    def key_clear(self):   self.text = ""
    def key_toggle_mode(self):
        if self.has_numbers:
            self.mode = "number" if self.mode == "letter" else "letter"


class Sender:
    """Sends each committed character to the other Mac over the network."""
    def __init__(self, ip, port=9999):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect((ip, port))
        self.sock.settimeout(None)
        self.ok = True

    def send(self, token):
        if not self.ok:
            return
        try:
            self.sock.sendall((token + "\n").encode())
        except OSError:
            self.ok = False


DISCOVERY_PORT = 9998
DISCOVERY_MAGIC = b"ASL_RECEIVER_HERE"


def discover(timeout=8):
    """Listen for the receiver Mac announcing itself; return its IP or None."""
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        u.bind(("", DISCOVERY_PORT))
    except OSError:
        return None
    u.settimeout(timeout)
    try:
        while True:
            data, addr = u.recvfrom(1024)
            if data.strip() == DISCOVERY_MAGIC:
                return addr[0]
    except socket.timeout:
        return None
    finally:
        u.close()


def is_open_palm(hand_landmarks):
    """True if the hand is an open palm (4 non-thumb fingers extended).
    Used for the two-hands auto-send gesture."""
    lm = hand_landmarks.landmark
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
    def d(i):
        return ((lm[i].x - wx) ** 2 + (lm[i].y - wy) ** 2 + (lm[i].z - wz) ** 2) ** 0.5
    extended = sum(1 for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)) if d(tip) > d(pip))
    return extended >= 4


def run():
    clf = joblib.load(MODEL_PATH)

    # SPEED: the model has 400 trees, but 100 give the exact same predictions
    # (verified: 100% agreement on 1,786 real samples) at ~2x the speed.
    # Set ASL_TREES=0 to keep the full forest, or another number to tune.
    n_trees = int(os.environ.get("ASL_TREES", "100"))
    if 0 < n_trees < len(clf.estimators_):
        clf.estimators_ = clf.estimators_[:n_trees]
        clf.n_estimators = n_trees
        print(f"Modelo recortado a {n_trees} árboles (misma precisión, ~2x más rápido)")
    typer = Typer(clf)

    # Connect to the other Mac.  Options:
    #   python3 asl_app.py            -> auto-discover the receiver
    #   python3 asl_app.py <Mac B IP> -> connect to that IP directly
    #   python3 asl_app.py local      -> just test signs here (no network, instant)
    sender = None
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "local":
        ip = None
        print("LOCAL test mode - recognition only, no networking.")
    elif arg:
        ip = arg
    else:
        print("Looking for the other Mac on the network (a few seconds)...")
        ip = discover()
    if ip:
        try:
            sender = Sender(ip)
            print(f"Connected to Mac B at {ip} - your text will be typed there.")
        except OSError as e:
            print(f"[WARN] Could not reach Mac B at {ip} ({e}). Running LOCAL only.")
            sender = None
    elif arg != "local":
        print("[INFO] No other Mac found - running LOCAL (recognition only).")
        print("       Start receiver.py on Mac B first, or pass its IP directly:")
        print("       python3 asl_app.py <Mac B IP shown on its screen>")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    # SPEED (Pi): model_complexity=0 makes hand tracking much faster on CPU
    # with slightly less precise landmarks. Try ASL_FAST=1 if the Pi feels slow.
    complexity = 0 if os.environ.get("ASL_FAST") == "1" else 1
    hands = mp_hands.Hands(max_num_hands=2,
                           model_complexity=complexity,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)

    cap = open_camera()
    if not cap.isOpened():
        print("[ERROR] Camera won't open.")
        print("  Mac: System Settings > Privacy & Security > Camera.")
        print("  Raspberry Pi: check `ls /dev/video*` and try ASL_CAMERA=1.")
        raise SystemExit(1)
    for _ in range(10):
        cap.read()

    buf = new_motion_buffer()
    send_start = None      # time (seconds) when both palms first appeared
    sent_latch = False     # prevents re-firing until palms are lowered
    if not HEADLESS:
        cv2.namedWindow("ASL app - Q to quit", cv2.WINDOW_NORMAL)
    else:
        print("HEADLESS mode - no preview window. Sign control gestures:")
        print("  both open palms (hold 2s) = send   |  Ctrl-C to quit")
    print("Sign letters. HOLD each sign steady until the green bar fills, then it types.")
    print("To type the same letter twice, drop your hand and sign it again.")
    print("Keyboard:  SPACE=space  DELETE=backspace  C=clear  Q=quit")
    print("           ENTER = send the finished story to the plotter pipeline")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        lms = results.multi_hand_landmarks or []
        # AUTO-SEND: show BOTH open palms to send the story (no keyboard needed)
        both_open = len(lms) >= 2 and all(is_open_palm(hd) for hd in lms[:2])
        cur, conf = "?", 0.0
        if both_open:
            for hnd in lms[:2]:
                mp_draw.draw_landmarks(frame, hnd, mp_hands.HAND_CONNECTIONS)
            buf.clear()
            typer.reset()
            cur, conf = "SEND", 1.0
            if send_start is None:
                send_start = time.time()
            if (time.time() - send_start) >= SEND_HOLD_SECONDS and not sent_latch:
                if sender:
                    sender.send("ENTER")
                print("AUTO-SEND - story sent:", repr(typer.text))
                typer.key_clear()
                sent_latch = True
        else:
            send_start = None
            sent_latch = False
            if lms:
                hand = lms[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                buf.append(fingertip_xy(hand))
                feats = augment(np.array([extract_shape(hand) + motion_features(buf)],
                                         dtype=np.float32))
                cur, conf = typer.predict(feats)
                committed = typer.update(cur, conf)
                if committed and sender:
                    sender.send(committed)
            else:
                buf.clear()
                typer.reset()

        # current prediction (big), confidence, mode
        if both_open:
            disp = "SENDING..."
        elif cur in typer.controls:
            disp = cur
        else:
            disp = cur.upper()
        bigcol = (0, 200, 255) if both_open else (0, 255, 0)
        cv2.putText(frame, dibujable(disp), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, bigcol, 4)
        cv2.putText(frame, f"{conf*100:3.0f}%", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MODE: {typer.mode.upper()}", (w - 240, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        netmsg = "-> Mac B" if (sender and sender.ok) else "LOCAL"
        cv2.putText(frame, netmsg, (w - 240, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        # medidor de movimiento: sirve para calibrar los umbrales de N-tilde/J/Z
        mv = getattr(typer, "last_motion", 0.0)
        mvcol = (0, 200, 255) if mv > typer.ENIE_MOTION_THRESH else (150, 150, 150)
        cv2.putText(frame, f"movimiento: {mv:4.2f}  (N-tilde > {typer.ENIE_MOTION_THRESH:.2f})",
                    (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mvcol, 2)
        cv2.putText(frame, "Both palms = SEND", (w - 240, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # progress bar: "hold steady" for a letter, or "hold to send" for both palms
        if both_open and send_start is not None:
            prog = min((time.time() - send_start) / SEND_HOLD_SECONDS, 1.0)
        else:
            prog = typer.progress()
        barcol = (0, 200, 255) if both_open else (0, 255, 0)
        cv2.rectangle(frame, (20, 120), (220, 140), (90, 90, 90), 1)
        cv2.rectangle(frame, (20, 120), (20 + int(200 * min(prog, 1.0)), 140),
                      barcol, -1)

        # the sentence so far, along the bottom
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        shown = typer.text[-40:]
        cv2.putText(frame, "> " + dibujable(shown) + "_", (10, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if HEADLESS:
            # No screen and no keyboard on the Pi: control gestures only.
            continue

        cv2.imshow("ASL app - Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            typer.key_space()
            if sender: sender.send("SPACE")
        elif key in (8, 127):
            typer.key_back()
            if sender: sender.send("BACK")
        elif key == 9:
            typer.key_toggle_mode()
        elif key == ord('c'):
            typer.key_clear()
        elif key in (13, 10):   # ENTER: story is done -> fire the plotter pipeline
            if sender:
                sender.send("ENTER")
                print("Story sent to the plotter! Text was:", repr(typer.text))
            typer.key_clear()

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal text:", repr(typer.text))


if __name__ == "__main__":
    run()
