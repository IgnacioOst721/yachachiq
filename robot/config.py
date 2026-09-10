"""Yachachiq configuration. Every other module reads its settings from here.

Edit this file, or override any value with an environment variable of the same
name prefixed with YACHACHIQ_ (example: YACHACHIQ_SERIAL_PORT=/dev/ttyUSB0).
Set YACHACHIQ_MOCK=1 to run the whole pipeline with no hardware, no models and
no laptop: the demo still listens, "draws" and narrates, so the UI can be tested
on any computer.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name, default):
    raw = os.environ.get("YACHACHIQ_" + name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, (list, tuple)):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(default, Path):
        return Path(raw).expanduser()
    return raw


# --- Global mode ---------------------------------------------------------------
MOCK = _env("MOCK", False)               # force every module into mock mode
OUTPUT_DIR = _env("OUTPUT_DIR", BASE_DIR / "output")

# --- Web UI / server ---------------------------------------------------------------
HOST = _env("HOST", "0.0.0.0")
PORT = _env("PORT", 8877)
TEXT_PORT = _env("TEXT_PORT", 5005)      # raw TCP port: one line of text = one story (gloves Pi)
GPIO_BUTTON_PIN = _env("GPIO_BUTTON_PIN", 17)   # optional physical button (see button.py)

# --- Audio ---------------------------------------------------------------------------
SAMPLE_RATE = 16000                       # Whisper wants 16 kHz mono
AUDIO_DEVICE = _env("AUDIO_DEVICE", "")   # "" = system default input; or a name/index for sounddevice
MAX_RECORD_SECONDS = _env("MAX_RECORD_SECONDS", 90.0)
SILENCE_RMS = _env("SILENCE_RMS", 0.012)  # RMS level (0..1) under which audio counts as silence
# After a story finishes, how long the result stays on screen before the welcome screen
# comes back ready for the next visitor. 0 = never reset on its own.
IDLE_RESET_SECONDS = _env("IDLE_RESET_SECONDS", 25.0)
NO_SPEECH_SECONDS = _env("NO_SPEECH_SECONDS", 10.0)  # nobody spoke at all -> back to start
SILENCE_SECONDS = _env("SILENCE_SECONDS", 3.5)   # auto-stop after this much trailing silence
MIN_SPEECH_SECONDS = _env("MIN_SPEECH_SECONDS", 1.0)   # ignore recordings shorter than this

# --- Speech to text (faster-whisper, runs on the Pi) -------------------------------
WHISPER_MODEL = _env("WHISPER_MODEL", "base")    # tiny / base / small / medium  (base: buen balance en la Pi 5)
WHISPER_COMPUTE = _env("WHISPER_COMPUTE", "int8")
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "es")  # "" = auto-detect (slower, less reliable)
WHISPER_CPU_THREADS = _env("WHISPER_CPU_THREADS", 4)

# --- Story analysis ----------------------------------------------------------------------
# Backends are tried in order until one succeeds. "rules" never fails.
#   remote     -> the laptop AI server (laptop/ai_server.py), Ollama behind it
#   ollama     -> talk to an Ollama instance directly
#   anthropic  -> Claude API (needs internet + ANTHROPIC_API_KEY)
#   rules      -> offline keyword rules, always available
STORY_BACKENDS = _env("STORY_BACKENDS", ["remote", "rules"])
# The laptop is reached by its mDNS name, which resolves both on a shared WiFi and over the
# direct Ethernet cable (avahi on the Pi, Bonjour on the Mac). Override with the env var if
# the Mac is renamed or a fixed IP is preferred (competition cable: 192.168.7.1).
LAPTOP_HOST = _env("LAPTOP_HOST", "el-loco-candy.local")
# At the competition there is no internet and no router: the Mac is wired straight to the Pi with
# static IPs (Mac 192.168.7.1). mDNS usually still resolves over that cable, but if it does not,
# every laptop request falls back to the fixed address instead of silently dropping to motifs.
LAPTOP_FALLBACK = _env("LAPTOP_FALLBACK", "192.168.7.1")


def laptop_urls(port):
    """Both ways to reach the laptop on a port, in order: mDNS name first, cable IP second."""
    hosts = [h for h in (LAPTOP_HOST, LAPTOP_FALLBACK) if h]
    seen, out = set(), []
    for h in hosts:
        if h not in seen:
            seen.add(h)
            out.append("http://%s:%d" % (h, port))
    return out

AI_SERVER_URL = _env("AI_SERVER_URL", f"http://{LAPTOP_HOST}:8600")
AI_SERVER_TIMEOUT = _env("AI_SERVER_TIMEOUT", 120.0)
OLLAMA_URL = _env("OLLAMA_URL", f"http://{LAPTOP_HOST}:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-opus-5")

# --- Image generation -------------------------------------------------------------------
#   comfyui -> ComfyUI on the laptop, runs laptop/comfyui_workflow.json with the story prompt patched in
#   remote  -> laptop AI server /generate (diffusers Stable Diffusion, no ComfyUI needed)
#   motifs  -> offline procedural Andean scene built from the story's elements (always works)
IMAGE_BACKENDS = _env("IMAGE_BACKENDS", ["comfyui", "remote", "motifs"])
# Tamano de la imagen que genera ComfyUI. Acepta "512" (cuadrada) o "512x720" (ancho x alto).
# Conviene que la proporcion se parezca a la del papel: con papel A4 vertical (1:1.41) una imagen
# cuadrada solo llena dos tercios de la hoja. 512x720 = 1:1.41, y SD 1.5 lo dibuja bien.
IMAGE_SIZE = _env("IMAGE_SIZE", "512x720")


def image_wh():
    """(ancho, alto) en pixeles a partir de IMAGE_SIZE."""
    v = str(IMAGE_SIZE).lower().replace(" ", "")
    if "x" in v:
        w, h = v.split("x", 1)
        return int(float(w)), int(float(h))
    return int(float(v)), int(float(v))
COMFYUI_URL = _env("COMFYUI_URL", f"http://{LAPTOP_HOST}:8188")
COMFYUI_WORKFLOW = _env("COMFYUI_WORKFLOW", BASE_DIR / "laptop" / "comfyui_workflow.json")   # API-format export
COMFYUI_CHECKPOINT = _env("COMFYUI_CHECKPOINT", "dreamshaper_8.safetensors")   # "" keeps the one in the workflow file
COMFYUI_STEPS = _env("COMFYUI_STEPS", 22)             # DreamShaper: 20-25 pasos (sd_turbo usa 4)
COMFYUI_CFG = _env("COMFYUI_CFG", 7.0)                # DreamShaper: ~7 (sd_turbo usa 1.0)
COMFYUI_TIMEOUT = _env("COMFYUI_TIMEOUT", 240.0)      # seconds to wait for the image
# The pen never lifts on this machine, so the picture itself is asked for as one-line art:
# fewer, longer strokes, and the few hops that remain look like part of the style.
ONE_LINE_STYLE = _env("ONE_LINE_STYLE", False)  # tested: one-line art loses the subject; coloring-book keeps it
STYLE_PROMPT = (
    # OUTLINES ONLY. Every filled area the model paints becomes dozens of plotter strokes, so the
    # prompt asks for a coloring-book page: thin clean contours, white inside, nothing shaded.
    ("(single continuous line drawing:1.4), one line art, minimalist, " if ONE_LINE_STYLE else "(coloring book style:1.3), ")
    + "(black outline drawing on white paper:1.3), (white background:1.4), no frame, no border, clean thin outlines only, "
    "no shading, no fill, no color, no texture, very few lines, bold simple shapes, centered, one scene, "
    "Peruvian Andean folk art style"
)
NEGATIVE_PROMPT = ("(black background:1.6), dark background, inverted colors, vignette, circle frame, (photograph:1.4), (photorealistic:1.4), grayscale photo, 3d render, color, colored, painting, shading, gradient, filled areas, solid black, silhouette, "
                   "hatching, crosshatch, texture, pattern, decorative border, frame, photo, realistic, "
                   "3d, blurry, text, watermark, noise, busy, cluttered, many objects")

# --- Image to lines (vectorize.py) ------------------------------------------------
TRACE_MODE = _env("TRACE_MODE", "lines")   # "lines" = threshold + thinning (for ink drawings); "edges" = Canny
TRACE_MAX_PX = _env("TRACE_MAX_PX", 600)   # working resolution (longest side)
CANNY_LOW = _env("CANNY_LOW", 60)
CANNY_HIGH = _env("CANNY_HIGH", 160)
# Pen always down: route the jumps between strokes back over lines already drawn (invisible)
# instead of straight across the paper; jumps shorter than this are just drawn.
CONTINUOUS_LINE = _env("CONTINUOUS_LINE", True)
JUMP_OK_MM = _env("JUMP_OK_MM", 3.0)
JOIN_TOL_MM = _env("JOIN_TOL_MM", 8.0)     # a stroke ending within 8 mm of another joins it (short bridge, no crossing)
# Filled areas in the image are drawn as ONE contour line instead of being hatched by the skeleton
# (a black shape used to become dozens of strokes). FILL_THICK_PX: a region thicker than this is a fill.
# Strokes that run around (almost) the whole picture are a decorative frame the model added,
# not part of the story: they are dropped (the pen spent most of its time on that border).
AI_BORDER = _env("AI_BORDER", False)          # decorative Andean border around AI drawings (off: only thin lines)
DROP_FRAME = _env("DROP_FRAME", True)
FRAME_MIN_COVER = _env("FRAME_MIN_COVER", 0.8)     # bbox wider AND taller than 80% of the image
FILL_TO_OUTLINE = _env("FILL_TO_OUTLINE", True)
FILL_THICK_PX = _env("FILL_THICK_PX", 8)   # thick outlines (< 16 px) become ONE centre line, only blobs get contours
SIMPLIFY_EPS_PX = _env("SIMPLIFY_EPS_PX", 0.6)   # Douglas-Peucker tolerance in pixels (0.6 = curvas mas fieles)
MIN_STROKE_PX = _env("MIN_STROKE_PX", 25.0)      # drop strokes shorter than this (pixels)
MAX_STROKES = _env("MAX_STROKES", 160)           # keep the longest N strokes. Measured with DreamShaper
                                                # line art: 350 ~= 9 min of plotting, 900 ~= 11 min.
                                                # Lower it (or raise MIN_STROKE_PX) for a faster demo.

# --- Plotter (Arduino UNO + CNC Shield + grbl-servo, over USB) ----------------------
SERIAL_PORT = _env("SERIAL_PORT", "/dev/ttyACM0")   # Pi: /dev/ttyACM0 or /dev/ttyUSB0; Mac: /dev/cu.usbmodem*
BAUD_RATE = 115200
PAPER_W_MM = _env("PAPER_W_MM", 15.0)     # ANCHO visto de frente = riel de arriba (GRBL Y): 15 mm pedidos = tope
PAPER_H_MM = _env("PAPER_H_MM", 24.0)     # ALTO visto de frente = riel izquierdo (GRBL X): 24 mm pedidos = tope
# On this machine GRBL's X axis is the LEFT (vertical) rail and Y is the TOP rail. Everything
# else in the code works in "as seen from the front" coordinates (width along the top rail);
# gcode.py swaps them when emitting, so drawings come out upright instead of rotated 90 degrees.
SWAP_XY = _env("SWAP_XY", True)
MARGIN_MM = _env("MARGIN_MM", 2.0)       # con 15 mm de ancho no cabe mas margen
DRAW_FEED = _env("DRAW_FEED", 100)        # mm/min PEDIDOS: esta maquina recorre ~10-20x lo pedido (sin calibrar)
TRAVEL_FEED = _env("TRAVEL_FEED", 150)    # mm/min pen up (pedidos)
PEN_FEED = _env("PEN_FEED", 300)          # mm/min Z moves
PEN_MODE = _env("PEN_MODE", "none")       # "none" = the pen NEVER lifts (Ignacio's machine): drawings are planned as one
                                          # continuous line. "z" = Z stepper lift, "servo" = grbl-servo M3/M5 on D11
PEN_UP_Z = _env("PEN_UP_Z", 15.0)      # el Z no esta calibrado: 3-4 mm pedidos no despegaban el plumon; 15 si
PEN_DOWN_Z = _env("PEN_DOWN_Z", 0.0)
SERVO_UP = _env("SERVO_UP", 90)           # M3 S<value> when PEN_MODE = servo
SERVO_DOWN = _env("SERVO_DOWN", 30)
# "Volver a casa" despues de cada dibujo: barrer hacia el tope izquierdo y el de abajo (la
# maquina no tiene finales de carrera) y fijar ahi el origen. Ignacio: ~100 mm izquierda, ~50 abajo.
HOME_SWEEP = _env("HOME_SWEEP", False)   # apagado: sin calibrar, 26 mm pedidos son 25-50 cm reales
HOME_SWEEP_X = _env("HOME_SWEEP_X", 26.0)   # recorrido X medido = 24 + margen
HOME_SWEEP_Y = _env("HOME_SWEEP_Y", 16.0)   # recorrido Y medido = 15 + margen
HOME_SWEEP_FEED = _env("HOME_SWEEP_FEED", 150)
MOCK_LINE_DELAY = _env("MOCK_LINE_DELAY", 0.004)   # seconds per G-code line in mock plotter (UI animation)

# --- Text to speech (Piper, runs on the Pi) ---------------------------------------
PIPER_VOICE = _env("PIPER_VOICE", Path.home() / ".local/share/piper/voices/es_MX-claude-high.onnx")
PIPER_BIN = _env("PIPER_BIN", "piper")
AUDIO_PLAYER = _env("AUDIO_PLAYER", "aplay")
SPEAK_WHILE_DRAWING = _env("SPEAK_WHILE_DRAWING", True)

# --- Drawing photo + web archive (Yachachiq additions) ----------------------------
PHOTO_ENABLED = _env("PHOTO_ENABLED", True)          # photograph the finished drawing
PHOTO_CAMERA = _env("PHOTO_CAMERA", "")              # "" = first camera, "1" = index, or /dev/v4l/by-id/... path
PHOTO_SETTLE_SECONDS = _env("PHOTO_SETTLE_SECONDS", 2.0)   # let the plotter stop shaking
STORIES_DIR = _env("STORIES_DIR", OUTPUT_DIR / "stories")  # one folder per story (text + images + photo)
PUBLISH_ENABLED = _env("PUBLISH_ENABLED", True)      # store-and-forward upload to the web gallery
# Stories are published from a separate clone of the `main` branch (GitHub Pages serves
# main:/docs), so publishing never mixes with the code branch the robot runs from.
# setup_pi.sh creates it at ~/yachachiq-archivo; without it, fall back to this repo.
_ARCHIVO = Path.home() / "yachachiq-archivo"
PUBLISH_REPO_DIR = Path(_env("PUBLISH_REPO_DIR", str(_ARCHIVO if _ARCHIVO.is_dir() else BASE_DIR.parent)))
LSP_CAMERA = _env("LSP_CAMERA", "")                  # sign-language camera for lsp_app (index or by-id path)

# --- Consent before publishing + storyteller portrait --------------------------------
CONSENT_REQUIRED = _env("CONSENT_REQUIRED", True)      # ask on screen before a story goes to the web
CONSENT_SECONDS = _env("CONSENT_SECONDS", 7)           # countdown while the portrait camera looks
PORTRAIT_ENABLED = _env("PORTRAIT_ENABLED", True)      # keep the storyteller's photo with the story
COVERED_BRIGHTNESS = _env("COVERED_BRIGHTNESS", 40.0)  # mean gray level below this = camera covered = "no"

# --- Hands-free start (no button, no touch, no keyboard) ----------------------------
# Hands-free start. Off by default since the welcome screen has a voice/signs chooser: with a
# real microphone the trigger fires on room noise and hijacks the robot while someone is signing.
AUTO_LISTEN = _env("AUTO_LISTEN", False)
AUTO_LISTEN_RMS = _env("AUTO_LISTEN_RMS", 0.030) # louder than SILENCE_RMS: a real voice, not room noise
AUTO_LISTEN_HOLD = _env("AUTO_LISTEN_HOLD", 0.35)# seconds of speech before it wakes up
AUTO_LISTEN_COOLDOWN = _env("AUTO_LISTEN_COOLDOWN", 3.0)  # pause after a story before listening again

# --- Story clean-up before drawing and publishing ------------------------------------
CLEAN_TEXT = _env("CLEAN_TEXT", True)        # sentence case, punctuation, safe accents
CLEAN_WITH_LLM = _env("CLEAN_WITH_LLM", True)  # also ask the laptop LLM to fix spelling
