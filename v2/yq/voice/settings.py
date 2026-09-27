"""Voice settings (Jetson side). Override any value with env YQ_<NAME>.

Example: YQ_MIC_NAME="SPEAKPHONE" YQ_VAD_SILENCE_S=1.5 python -m yq.voice.capture --test
"""
from __future__ import annotations

from pathlib import Path

from yq.common import config
from yq.common.config import env

SAMPLE_RATE = 16000                                   # every model here wants 16 kHz mono

# --- Microphone / speaker (KAYSUDA USB speakerphone) ---------------------------
# Matched as a case-insensitive substring of the sounddevice device name, in
# order; the first match wins, else the system default device is used. On
# Linux a USB speakerphone shows up as "<USB product>: USB Audio (hw:N,0)".
MIC_NAMES = env("MIC_NAMES", ["KAYSUDA", "SPEAKPHONE", "SP300", "SP200", "Speakerphone", "USB Audio"])
SPEAKER_NAMES = env("SPEAKER_NAMES", ["KAYSUDA", "SPEAKPHONE", "SP300", "SP200", "Speakerphone", "USB Audio"])
MIC_DEVICE = env("MIC_DEVICE", "")                    # exact index or name; overrides MIC_NAMES
SPEAKER_DEVICE = env("SPEAKER_DEVICE", "")
MIC_BLOCK_MS = env("MIC_BLOCK_MS", 32)                # 512 samples at 16 kHz = one Silero VAD frame

# --- Voice activity detection (Silero VAD, ONNX) --------------------------------
VAD_MODEL = env("VAD_MODEL", "")                       # "" = the copy shipped in yq/voice/data (v6.2.3, MIT)
VAD_MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx"
VAD_THRESHOLD = env("VAD_THRESHOLD", 0.5)             # speech starts when prob >= this
VAD_NEG_THRESHOLD = env("VAD_NEG_THRESHOLD", 0.35)    # and ends when prob < this (Silero: threshold - 0.15)
VAD_MIN_SPEECH_S = env("VAD_MIN_SPEECH_S", 0.25)      # shorter bursts (a cough, a click) do not start a story
VAD_SILENCE_S = env("VAD_SILENCE_S", 1.8)             # this much silence after speech = the visitor finished
VAD_PREROLL_S = env("VAD_PREROLL_S", 0.5)             # audio kept from before speech started
VAD_POSTROLL_S = env("VAD_POSTROLL_S", 0.3)           # audio kept after the last speech
NO_SPEECH_TIMEOUT_S = env("NO_SPEECH_TIMEOUT_S", 12.0)  # nobody spoke: give up (returns empty audio)
MAX_RECORD_S = env("MAX_RECORD_S", 90.0)
# Energy fallback when the ONNX model is missing (much worse than Silero; logged loudly)
ENERGY_FACTOR = env("ENERGY_FACTOR", 3.0)             # speech = RMS > factor x noise floor
ENERGY_MIN_RMS = env("ENERGY_MIN_RMS", 0.01)

# --- Speech to text ------------------------------------------------------------------
ASR_MAC_TIMEOUT_S = env("ASR_MAC_TIMEOUT_S", 120.0)   # long story + model load on the Mac
ASR_LOCAL = env("ASR_LOCAL", True)                    # try the Jetson model when the Mac fails
# faster-whisper on the Jetson GPU (CTranslate2 CUDA build) and a smaller CPU fallback
ASR_JETSON_MODEL = env("ASR_JETSON_MODEL", "dropbox-dash/faster-whisper-large-v3-turbo")   # MIT; the "large-v3-turbo" alias redirects here
ASR_JETSON_COMPUTE = env("ASR_JETSON_COMPUTE", "int8_float16")
ASR_CPU_MODEL = env("ASR_CPU_MODEL", "small")
ASR_CPU_COMPUTE = env("ASR_CPU_COMPUTE", "int8")
ASR_CPU_THREADS = env("ASR_CPU_THREADS", 4)
ASR_LOCAL_TIMEOUT_S = env("ASR_LOCAL_TIMEOUT_S", 180.0)
ASR_BEAM = env("ASR_BEAM", 5)
WHISPER_CACHE = env("WHISPER_CACHE", "")               # "" = MODELS_DIR/whisper (faster-whisper download_root)

# --- Text to speech ------------------------------------------------------------------
PIPER_DIR = env("PIPER_DIR", "")                       # "" = MODELS_DIR/piper
PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
PIPER_USE_CUDA = env("PIPER_USE_CUDA", False)         # piper is fast enough on the Orin CPU
PIPER_VOICE_OVERRIDES = env("PIPER_VOICE_OVERRIDES", ["spa_Latn=es_MX-claude-high", "eng_Latn=en_US-lessac-high"])
TTS_MAC_TIMEOUT_S = env("TTS_MAC_TIMEOUT_S", 60.0)
TTS_LENGTH_SCALE = env("TTS_LENGTH_SCALE", 1.05)      # >1 = a little slower, clearer for visitors
AUDIO_PLAYER = env("AUDIO_PLAYER", "")                # "" = play with sounddevice; or e.g. "aplay"

# --- Translation ---------------------------------------------------------------------
TRANSLATE_TIMEOUT_S = env("TRANSLATE_TIMEOUT_S", 60.0)


def piper_overrides() -> dict:
    out = {}
    for item in PIPER_VOICE_OVERRIDES:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def vad_model_path() -> Path:
    return Path(VAD_MODEL).expanduser() if VAD_MODEL else Path(__file__).resolve().parent / "data" / "silero_vad.onnx"


def piper_dir() -> Path:
    return Path(PIPER_DIR).expanduser() if PIPER_DIR else Path(config.MODELS_DIR) / "piper"


def whisper_cache() -> Path:
    return Path(WHISPER_CACHE).expanduser() if WHISPER_CACHE else Path(config.MODELS_DIR) / "whisper"

# --- Mac worker (routes_voice.py) ------------------------------------------------------
# Which models answer /asr on the Mac (see docs/voice.md for the measurements behind the defaults)
MAC_WHISPER_MODEL = env("MAC_WHISPER_MODEL", "whisper-large-v3-turbo")   # measured = large-v3 accuracy, 2.5x faster, half the memory
# known language (the visitor picked it): the LLM variant is far more accurate on Quechua (Puno WER 10.0 vs
# 32.5 with CTC 1B, 2026-09-27) at RTF ~0.5 and 5.9 GB; CTC 1B stays as its fallback and for "auto"
MAC_OMNI_MODEL_KNOWN = env("MAC_OMNI_MODEL_KNOWN", "omniasr-llm-1b")
MAC_OMNI_MODEL = env("MAC_OMNI_MODEL", "omniasr-ctc-1b")              # or "omniasr-llm-1b", "omniasr-ctc-300m"
MAC_MMS_LID = env("MAC_MMS_LID", True)                                # MMS-LID for languages Whisper lacks
LID_WHISPER_MIN = env("LID_WHISPER_MIN", 0.60)    # trust Whisper's language guess above this (strong languages)
LID_MMS_MIN = env("LID_MMS_MIN", 0.50)            # trust MMS-LID's guess above this
ASR_RETRY_MIN_CONF = env("ASR_RETRY_MIN_CONF", 0.35)   # below this, try the next engine and keep the best
# Python that runs Omnilingual ASR in its child process. "" = v2/.venvs/voice/bin/python when it
# exists (keeps fairseq2/torch pins out of the worker's own venv), else the worker's Python.
VOICE_PYTHON = env("VOICE_PYTHON", "")


def voice_python() -> str:
    import sys
    if VOICE_PYTHON:
        return VOICE_PYTHON
    cand = config.V2_DIR / ".venvs" / "voice" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable
