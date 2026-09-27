"""SIGN settings (env YQ_<NAME>, see yq.common.config.env)."""
from __future__ import annotations

from yq.common.config import env

# Camera (Arducam OV9782 USB: native 1280x800, MJPG up to 100 fps; no 1080p mode)
SIGN_CAMERA = env("SIGN_CAMERA", "0")             # index or /dev/v4l/by-id/... path
SIGN_CAM_WIDTH = env("SIGN_CAM_WIDTH", 1280)
SIGN_CAM_HEIGHT = env("SIGN_CAM_HEIGHT", 800)
SIGN_CAM_FPS = env("SIGN_CAM_FPS", 60)
SIGN_MIRROR = env("SIGN_MIRROR", True)            # preview like a mirror (the model sees the same frame)

# Pose
SIGN_POSE_MODEL = env("SIGN_POSE_MODEL", "rtmw-m")   # rtmw-m | rtmw-l | rtmw-l-384
SIGN_BACKEND = env("SIGN_BACKEND", "auto")            # auto | tensorrt | cuda | coreml | cpu
SIGN_DETECTOR = env("SIGN_DETECTOR", "yolox-tiny")    # "" = no detector (whole frame)
SIGN_REDETECT_S = env("SIGN_REDETECT_S", 2.0)          # re-run the person detector at least this often
SIGN_HAND_REFINE = env("SIGN_HAND_REFINE", "letters")  # off | letters | always (RTMPose-m hand crop)
SIGN_KPT_THR = env("SIGN_KPT_THR", 0.3)

# One Euro filter (coordinates in pixels, speed divided by the person box height).
# Defaults = MediaPipe's pose landmark smoothing (mediapipe/modules/pose_landmark/
# pose_landmark_filtering.pbtxt: min_cutoff 0.05, beta 80, derivate_cutoff 1.0, speed scaled by
# the object size): very smooth when still, almost no lag when moving.
SIGN_EURO_MIN_CUTOFF = env("SIGN_EURO_MIN_CUTOFF", 0.05)  # Hz: lower = smoother when still
SIGN_EURO_BETA = env("SIGN_EURO_BETA", 80.0)              # higher = less lag when moving fast
SIGN_EURO_D_CUTOFF = env("SIGN_EURO_D_CUTOFF", 1.0)

# Letters
SIGN_HOLD_S = env("SIGN_HOLD_S", 0.30)          # hold a letter this long (hand still) to write it (v1: 0.30)
SIGN_LETTER_MIN_CONF = env("SIGN_LETTER_MIN_CONF", 0.55)
SIGN_STILL_SPEED = env("SIGN_STILL_SPEED", 2.0)  # palm units per second: slower = "still"
SIGN_AUTO_SPACE_S = env("SIGN_AUTO_SPACE_S", 1.5)  # hand down this long after a word = space

# Words
SIGN_WORD_MIN_CONF = env("SIGN_WORD_MIN_CONF", 0.35)
SIGN_WORD_REST_S = env("SIGN_WORD_REST_S", 0.45)   # hands still/down this long = end of a sign
SIGN_WORD_MAX_S = env("SIGN_WORD_MAX_S", 4.0)

SIGN_PREVIEW_FPS = env("SIGN_PREVIEW_FPS", 15.0)
SIGN_PREVIEW_WIDTH = env("SIGN_PREVIEW_WIDTH", 640)
