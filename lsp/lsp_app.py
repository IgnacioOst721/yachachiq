import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import sys
import socket
import threading
import time
from collections import deque
from landmark_utils import (extract_shape, fingertip_xy, motion_features,
                            new_motion_buffer, augment, open_camera)


# OpenCV solo dibuja ASCII, asi que la Ñ le sale como "?". Para escribirla
# bien se usa PIL con una fuente real, pero SOLO cuando hace falta: el texto
# normal se sigue dibujando con cv2, que es mucho mas rapido (importa en la Pi).
_FUENTES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",   # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",           # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Raspberry Pi
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_cache_fuente = {}


def _fuente(px):
    if px not in _cache_fuente:
        from PIL import ImageFont
        f = None
        for ruta in _FUENTES:
            if os.path.exists(ruta):
                try:
                    f = ImageFont.truetype(ruta, px); break
                except Exception:
                    pass
        _cache_fuente[px] = f or ImageFont.load_default()
    return _cache_fuente[px]


def texto(frame, msg, pos, escala, color, grosor):
    """Dibuja texto. Usa cv2 si es ASCII; PIL si trae Ñ u otro acento."""
    if msg.isascii():
        cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor)
        return frame
    from PIL import Image, ImageDraw
    px = max(12, int(escala * 30))
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text((pos[0], pos[1] - px), msg,
                             font=_fuente(px), fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame


def dibujable(t):
    return t

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
STABLE_FRAMES = 3      # minimo de frames iguales seguidos (piso de seguridad)
# Una letra se acepta cuando la MISMA prediccion se mantiene este tiempo. Por tiempo y
# no por frames, asi en la Pi (menos fps) no se vuelve mas lento. Ajustable sin codigo.
STABLE_SECONDS = float(os.environ.get("LSP_STABLE_SECONDS", "0.30"))
# Si la mano sale del cuadro este tiempo, se mete un espacio solo (no hace falta la
# seña SPACE entre palabras). 0 = desactivado.
AUTO_SPACE_SECONDS = float(os.environ.get("LSP_AUTO_SPACE", "1.2"))
CONF_THRESH = 0.60     # minimum confidence to accept a sign (0.60: fewer wrong letters)
VOTE_WINDOW = 5        # frames considered for the majority vote
VOTE_MIN = 3           # votes a letter needs to be shown/held
SEND_HOLD_SECONDS = 2.0  # hold BOTH open palms this long to auto-send

# ── ROBOT (pipeline de Yachachiq en la misma Pi) ────────────────────────
# La historia señada se manda al servidor del robot (server.py) que la dibuja,
# y la camara se transmite a la pantalla del kiosko para que se vean las señas.
#   LSP_ROBOT=http://127.0.0.1:8877   (""  = desactivado)
#   LSP_ROBOT_TEXT_PORT=5005          puerto TCP de texto de server.py
LSP_ROBOT = os.environ.get("LSP_ROBOT", "http://127.0.0.1:8877").rstrip("/")
LSP_ROBOT_TEXT_PORT = int(os.environ.get("LSP_ROBOT_TEXT_PORT", "5005"))


def _robot_host():
    from urllib.parse import urlparse
    return urlparse(LSP_ROBOT).hostname or "127.0.0.1"


def robot_send_story(text):
    """Manda la historia completa al robot (una linea por TCP). True si la acepto."""
    text = " ".join(text.split()).strip()
    if not LSP_ROBOT or not text:
        return False
    try:
        with socket.create_connection((_robot_host(), LSP_ROBOT_TEXT_PORT), timeout=5) as c:
            c.sendall((text + "\n").encode("utf-8"))
            resp = c.recv(64).decode(errors="ignore").strip()
        ok = resp.startswith("ok")
        print(("ROBOT: historia enviada -> dibujando" if ok else "ROBOT ocupado, no acepto la historia"), repr(text))
        return ok
    except OSError as e:
        print(f"[ROBOT] no se pudo enviar ({e}); ¿esta corriendo server.py?")
        return False


def robot_notice(text):
    """Show a sentence to the visitor on the kiosk (best effort, never raises)."""
    if not LSP_ROBOT:
        return
    try:
        import json as _json
        import urllib.request
        req = urllib.request.Request(LSP_ROBOT + "/api/notice", data=_json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2).read()
    except Exception:
        pass


class RobotState:
    """Follows the robot's state so hand tracking only runs when somebody is actually signing.

    MediaPipe costs a full CPU core; on a Raspberry Pi 5 leaving it on all day pushed the chip to
    85 C and the whole machine throttled. While the robot is drawing, imagining or idle we keep
    sending the camera picture (the kiosk and the consent portrait need it) but skip the tracking.
    """
    TRACK_IN = ("waiting_signs",)          # hand tracking (expensive)
    SHOW_IN = ("waiting_signs", "consent")  # the kiosk is displaying the camera

    def __init__(self):
        self.state = "idle"
        self.enabled = bool(LSP_ROBOT)
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()

    def tracking(self):
        return (not self.enabled) or self.state in self.TRACK_IN

    def showing(self):
        return (not self.enabled) or self.state in self.SHOW_IN

    def _loop(self):
        import json as _json
        import urllib.request
        while True:
            try:
                with urllib.request.urlopen(LSP_ROBOT + "/api/state", timeout=2) as r:
                    self.state = _json.loads(r.read().decode()).get("state", self.state)
            except Exception:
                self.state = "waiting_signs"     # cannot ask: assume someone may be signing
            time.sleep(1.5)


class RobotPreview:
    """Envia al kiosko un frame pequeño ~12 veces por segundo y el texto actual (hilo aparte).
    El kiosko lo muestra como stream MJPEG continuo, sin polling."""
    def __init__(self):
        self.frame = None
        self.seq = 0
        self._sent = -1
        self.text = ("", "", "letter")
        self._last_text = None
        self.enabled = bool(LSP_ROBOT)
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()

    def update(self, frame, text=None, letter=None, mode=None):
        # copy now: the main loop draws its debug HUD onto `frame` right after this call, and the
        # kiosk (and the consent portrait) must get the clean picture. The text is only replaced
        # when given: sending letter="" with every frame made the kiosk flicker between the
        # recognised letter and a dot.
        self.frame = frame.copy()
        self.seq += 1
        if text is not None:
            self.text = (text, letter or "", mode or "letter")

    def _post(self, path, data, ctype):
        # one persistent HTTP connection instead of a new TCP handshake per frame
        import http.client
        from urllib.parse import urlparse
        if getattr(self, "_conn", None) is None:
            u = urlparse(LSP_ROBOT)
            self._conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=1.5)
        try:
            self._conn.request("POST", path, body=data, headers={"Content-Type": ctype})
            self._conn.getresponse().read()
        except Exception:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            raise

    def _loop(self):
        import json
        fails = 0
        while True:
            time.sleep(0.04)                      # up to 25 fps; only NEW frames are sent
            try:
                if self.frame is not None and self.seq != self._sent:
                    self._sent = self.seq
                    small = cv2.resize(self.frame, (400, int(self.frame.shape[0] * 400 / self.frame.shape[1])),
                                       interpolation=cv2.INTER_AREA)
                    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 55])
                    if ok:
                        self._post("/api/lsp/frame", jpg.tobytes(), "image/jpeg")
                if self.text != self._last_text:
                    t, l, m = self.text
                    self._post("/api/lsp/text", json.dumps({"text": t, "letter": l, "mode": m}).encode(), "application/json")
                    self._last_text = self.text
                fails = 0
            except Exception:
                fails += 1
                if fails in (1, 50):
                    print("[ROBOT] kiosko no responde en", LSP_ROBOT, "(sigo intentando)")
                time.sleep(2)


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
        self._stable_since = None  # cuando empezo la prediccion actual
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

        # Z solo vale si la mano se esta moviendo. Con la mano quieta,
        # pasar a la mejor letra que no sea de movimiento.
        if best == "z" and total < self.MIN_MOTION_JZ:
            for c in ordenadas[1:]:
                if c != "z":
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

    def update(self, cur, conf, now=None):
        """Feed one frame. Returns the committed label, or None if nothing fired.
        A sign counts when it has held for STABLE_SECONDS (and >= stable_frames frames)."""
        now = time.time() if now is None else now
        if cur == self._prev:
            self._stable += 1
        else:
            self._stable = 1
            self._stable_since = now
            self._prev = cur
        held = now - (self._stable_since if self._stable_since is not None else now)
        if (self._stable >= self.stable_frames and held >= STABLE_SECONDS
                and conf >= self.conf_thresh and cur != self._committed and cur != "?"):
            self._committed = cur
            self._apply(cur)
            return cur
        return None

    def reset(self):
        """Call when the hand leaves the frame, so the same sign can repeat."""
        self._prev = None
        self._stable = 0
        self._stable_since = None
        self._committed = None
        self._mov_hist.clear()

    def progress(self):
        if self._stable_since is None or STABLE_SECONDS <= 0:
            return 0.0
        return min((time.time() - self._stable_since) / STABLE_SECONDS, 1.0)

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
    elif arg == "robot":
        ip = None
        print(f"ROBOT mode - las historias van al robot en {LSP_ROBOT} (puerto texto {LSP_ROBOT_TEXT_PORT}).")
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
    elif arg not in ("local", "robot"):
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
                           min_tracking_confidence=0.5)   # steadier tracking between frames, fewer dropouts

    cap = open_camera()
    # As a background service on the Pi (ASL_HEADLESS=1) the webcam may simply not be
    # plugged in yet: wait for it instead of exiting, so systemd does not restart us
    # (and reload MediaPipe) every few seconds.
    while not cap.isOpened():
        if os.environ.get("ASL_HEADLESS") != "1":
            print("[ERROR] Camera won't open.")
            print("  Mac: System Settings > Privacy & Security > Camera.")
            print("  Raspberry Pi: check `ls /dev/video*` and try ASL_CAMERA=1.")
            raise SystemExit(1)
        print("[LSP] no camera yet - retrying in 15 s (plug in the sign webcam)", flush=True)
        cap.release(); time.sleep(15); cap = open_camera()
    for _ in range(10):
        cap.read()

    preview = RobotPreview()
    robot = RobotState()
    hand_gone_since = None     # para el espacio automatico entre palabras
    votes = []                 # (letra, confianza) de los ultimos frames
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

    was_tracking = False
    read_fails = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # USB camera dropout: this used to spin at 100% CPU forever with a frozen picture.
            # Now: wait a little, re-open the camera every ~5 s, and after a minute let systemd
            # restart the whole app (Restart=always).
            read_fails += 1
            if read_fails == 1:
                print("[LSP] la camara no entrega imagen; reintentando", flush=True)
            time.sleep(0.05)
            if read_fails % 100 == 0:
                cap.release()
                cap = open_camera()
            if read_fails > 1200:
                print("[LSP] la camara no volvio en un minuto: saliendo para que el servicio reinicie", flush=True)
                raise SystemExit(3)
            continue
        read_fails = 0
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        tracking = robot.tracking()
        # consent (no hand tracking): the clean picture goes to the kiosk right away. While signing,
        # ONE frame per loop goes out below, with the hand dots drawn on it: sending a clean frame
        # here and a dotted one later made the dots flicker on and off.
        if robot.showing() and not tracking:
            preview.update(frame)
        if was_tracking and not tracking:
            # the visitor left sign mode (story sent, or chose another way): forget the half-typed
            # text so it is not glued to the front of the next person's story
            typer.key_clear()
            typer.reset()
            votes.clear()
            buf.clear()
            send_start, sent_latch, hand_gone_since = None, False, None
            preview.text = ("", "", typer.mode)
        was_tracking = tracking
        if not tracking:
            # nobody is signing: skip hand tracking entirely
            if not HEADLESS:
                cv2.imshow("ASL app - Q to quit", frame)
                if cv2.waitKey(60) & 0xFF in (ord("q"), 27):
                    break
            else:
                time.sleep(0.08 if robot.showing() else 0.4)
            continue

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
                sent_latch = True
                if not typer.text.strip():
                    robot_notice("Todavía no hay letras. Seña tu historia letra por letra y luego muestra las dos palmas.")
                elif robot_send_story(typer.text):
                    print("AUTO-SEND - story sent:", repr(typer.text))
                    typer.key_clear()
                else:
                    # the robot was busy or unreachable: KEEP the text (it used to be wiped and the
                    # signed story was silently lost) and tell the visitor what to do
                    robot_notice("El robot no pudo recibir tu historia todavía. Espera a que termine "
                                 "y muestra las dos palmas otra vez; tu texto sigue aquí.")
        else:
            send_start = None
            sent_latch = False
            if lms:
                hand = lms[0]
                # the visitor sees the dots on their own hand: without them there is no way to
                # tell "the robot does not see my hand" from "it sees it and does not know the sign"
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                buf.append(fingertip_xy(hand))
                feats = augment(np.array([extract_shape(hand) + motion_features(buf)],
                                         dtype=np.float32))
                cur, conf = typer.predict(feats)
                # majority vote over the last few frames: one mis-read frame no longer resets
                # the hold timer or sneaks a wrong letter through
                votes.append((cur, conf))
                if len(votes) > VOTE_WINDOW:
                    votes.pop(0)
                if len(votes) >= 3:
                    best = max(set(v for v, _ in votes), key=lambda L: sum(1 for v, _ in votes if v == L))
                    n = sum(1 for v, _ in votes if v == best)
                    if n >= VOTE_MIN:
                        cur = best
                        conf = sum(c for v, c in votes if v == best) / n
                    else:
                        cur, conf = "?", 0.0
                committed = typer.update(cur, conf)
                if committed and sender:
                    sender.send(committed)
            else:
                buf.clear()
                typer.reset()
                # ESPACIO AUTOMATICO: si la mano sale del cuadro un momento, es fin de
                # palabra. Bajar la mano y volver rapido (< AUTO_SPACE_SECONDS) sigue
                # sirviendo para repetir una letra sin meter espacio.
                if hand_gone_since is None:
                    hand_gone_since = time.time()
                elif (AUTO_SPACE_SECONDS > 0 and typer.text and not typer.text.endswith(" ")
                      and time.time() - hand_gone_since >= AUTO_SPACE_SECONDS):
                    typer.key_space()
                    if sender: sender.send("SPACE")
        if lms:
            hand_gone_since = None
        if robot.showing():
            preview.update(frame)          # one frame per loop, dots included (HUD is drawn after this)

        preview.text = (typer.text, ("SEND" if both_open else (cur.upper() if len(cur) == 1 else cur)), typer.mode)

        if HEADLESS:
            # No screen and no keyboard on the Pi: the HUD below would be drawn for nobody
            # (and with an Ñ on screen it cost a full-frame PIL round trip per frame).
            continue

        # current prediction (big), confidence, mode
        if both_open:
            disp = "SENDING..."
        elif cur in typer.controls:
            disp = cur
        else:
            disp = cur.upper()
        bigcol = (0, 200, 255) if both_open else (0, 255, 0)
        texto(frame, disp, (20, 70), 1.6, bigcol, 4)
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
        texto(frame, "> " + shown + "_", (10, h - 22), 0.9, (255, 255, 255), 2)

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
            robot_send_story(typer.text)
            print("Story sent to the plotter! Text was:", repr(typer.text))
            typer.key_clear()

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal text:", repr(typer.text))


if __name__ == "__main__":
    run()
