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
AI_SERVER_URL = _env("AI_SERVER_URL", "http://192.168.7.1:8600")
AI_SERVER_TIMEOUT = _env("AI_SERVER_TIMEOUT", 120.0)
OLLAMA_URL = _env("OLLAMA_URL", "http://192.168.7.1:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-opus-5")

# --- Image generation -------------------------------------------------------------------
#   comfyui -> ComfyUI on the laptop, runs laptop/comfyui_workflow.json with the story prompt patched in
#   remote  -> laptop AI server /generate (diffusers Stable Diffusion, no ComfyUI needed)
#   motifs  -> offline procedural Andean scene built from the story's elements (always works)
IMAGE_BACKENDS = _env("IMAGE_BACKENDS", ["comfyui", "remote", "motifs"])
IMAGE_SIZE = _env("IMAGE_SIZE", 512)
COMFYUI_URL = _env("COMFYUI_URL", "http://192.168.7.1:8188")
COMFYUI_WORKFLOW = _env("COMFYUI_WORKFLOW", BASE_DIR / "laptop" / "comfyui_workflow.json")   # API-format export
COMFYUI_CHECKPOINT = _env("COMFYUI_CHECKPOINT", "dreamshaper_8.safetensors")   # "" keeps the one in the workflow file
COMFYUI_STEPS = _env("COMFYUI_STEPS", 22)             # DreamShaper: 20-25 pasos (sd_turbo usa 4)
COMFYUI_CFG = _env("COMFYUI_CFG", 7.0)                # DreamShaper: ~7 (sd_turbo usa 1.0)
COMFYUI_TIMEOUT = _env("COMFYUI_TIMEOUT", 240.0)      # seconds to wait for the image
STYLE_PROMPT = (
    "black ink line drawing, Peruvian folk art, Sarhua tabla style, Ayacucho retablo, "
    "thick clean outlines, flat 2D, decorative Andean geometric border, white background, "
    "no shading, no color, single continuous lines, simple shapes"
)
NEGATIVE_PROMPT = "color, shading, gradient, photo, realistic, blurry, text, watermark, noise"

# --- Image to lines (vectorize.py) ------------------------------------------------
TRACE_MODE = _env("TRACE_MODE", "lines")   # "lines" = threshold + thinning (for ink drawings); "edges" = Canny
TRACE_MAX_PX = _env("TRACE_MAX_PX", 600)   # working resolution (longest side)
CANNY_LOW = _env("CANNY_LOW", 60)
CANNY_HIGH = _env("CANNY_HIGH", 160)
SIMPLIFY_EPS_PX = _env("SIMPLIFY_EPS_PX", 1.2)   # Douglas-Peucker tolerance in pixels
MIN_STROKE_PX = _env("MIN_STROKE_PX", 20.0)      # drop strokes shorter than this (pixels)
MAX_STROKES = _env("MAX_STROKES", 350)           # keep the longest N strokes. Measured with DreamShaper
                                                # line art: 350 ~= 9 min of plotting, 900 ~= 11 min.
                                                # Lower it (or raise MIN_STROKE_PX) for a faster demo.

# --- Plotter (Arduino UNO + CNC Shield + grbl-servo, over USB) ----------------------
SERIAL_PORT = _env("SERIAL_PORT", "/dev/ttyACM0")   # Pi: /dev/ttyACM0 or /dev/ttyUSB0; Mac: /dev/cu.usbmodem*
BAUD_RATE = 115200
PAPER_W_MM = _env("PAPER_W_MM", 210.0)    # A4 landscape by default
PAPER_H_MM = _env("PAPER_H_MM", 148.0)    # half A4 height keeps plots short; set 297 for portrait A4
MARGIN_MM = _env("MARGIN_MM", 12.0)
DRAW_FEED = _env("DRAW_FEED", 800)        # mm/min pen down
TRAVEL_FEED = _env("TRAVEL_FEED", 1500)   # mm/min pen up
PEN_FEED = _env("PEN_FEED", 300)          # mm/min Z moves
PEN_MODE = _env("PEN_MODE", "z")          # "z" = Z stepper lift, "servo" = grbl-servo M3/M5 on D11
PEN_UP_Z = _env("PEN_UP_Z", 5.0)
PEN_DOWN_Z = _env("PEN_DOWN_Z", 0.0)
SERVO_UP = _env("SERVO_UP", 90)           # M3 S<value> when PEN_MODE = servo
SERVO_DOWN = _env("SERVO_DOWN", 30)
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
PUBLISH_REPO_DIR = _env("PUBLISH_REPO_DIR", BASE_DIR.parent)   # git repo that holds docs/ (GitHub Pages)
LSP_CAMERA = _env("LSP_CAMERA", "")                  # sign-language camera for lsp_app (index or by-id path)

# --- Consent before publishing + storyteller portrait --------------------------------
CONSENT_REQUIRED = _env("CONSENT_REQUIRED", True)      # ask on screen before a story goes to the web
CONSENT_SECONDS = _env("CONSENT_SECONDS", 7)           # countdown while the portrait camera looks
PORTRAIT_ENABLED = _env("PORTRAIT_ENABLED", True)      # keep the storyteller's photo with the story
COVERED_BRIGHTNESS = _env("COVERED_BRIGHTNESS", 40.0)  # mean gray level below this = camera covered = "no"

# --- Hands-free start (no button, no touch, no keyboard) ----------------------------
AUTO_LISTEN = _env("AUTO_LISTEN", True)          # start recording by itself when someone speaks
AUTO_LISTEN_RMS = _env("AUTO_LISTEN_RMS", 0.030) # louder than SILENCE_RMS: a real voice, not room noise
AUTO_LISTEN_HOLD = _env("AUTO_LISTEN_HOLD", 0.35)# seconds of speech before it wakes up
AUTO_LISTEN_COOLDOWN = _env("AUTO_LISTEN_COOLDOWN", 3.0)  # pause after a story before listening again

# --- Story clean-up before drawing and publishing ------------------------------------
CLEAN_TEXT = _env("CLEAN_TEXT", True)        # sentence case, punctuation, safe accents
CLEAN_WITH_LLM = _env("CLEAN_WITH_LLM", True)  # also ask the laptop LLM to fix spelling
