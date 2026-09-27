"""Settings of the analysis box capture side (BOX-CAPTURE).

Every value can be overridden with an environment variable YQ_BOX_<NAME>
(see yq.common.config.env). Geometry defaults come from the CAD R1 tables
(Tablas/componentes.json, CAD/parametros.json) converted to the object frame of
CONTRACTS.md §4: origin = platter centre on the platter top surface, X right,
Y towards the back wall, Z up, millimetres. CAD box frame -> object frame:
subtract (150, 150, 82).
"""
from __future__ import annotations

import math
from pathlib import Path

from yq.common import config
from yq.common.config import env

# --- Serial link to the ESP32 --------------------------------------------------------
SERIAL_PORT = env("BOX_SERIAL_PORT", "")          # "" = auto discovery
SERIAL_BAUD = env("BOX_SERIAL_BAUD", 115200)
# Substrings that identify the DevKitC's USB-UART bridge (CP2102N on genuine
# boards, CH340/CH9102 on clones) in /dev/serial/by-id or /dev/cu.* names.
SERIAL_HINTS = env("BOX_SERIAL_HINTS", ["CP210", "Silicon_Labs", "SLAB_USBtoUART", "CH340", "CH910",
                                        "wchusbserial", "usbserial", "1a86"])
REQUEST_TIMEOUT_S = env("BOX_REQUEST_TIMEOUT_S", 2.0)
REQUEST_RETRIES = env("BOX_REQUEST_RETRIES", 2)
HEARTBEAT_S = env("BOX_HEARTBEAT_S", 1.0)         # firmware cuts everything after 3 s of silence
BOOT_WAIT_S = env("BOX_BOOT_WAIT_S", 2.5)         # opening the port resets the DevKitC (DTR/RTS)

# --- Firmware mirror (must match firmware/box_esp32/include/{pins,boxconfig}.h) --------
PINS = {"step": 25, "dir": 26, "en": 27, "tmc_tx": 17, "tmc_rx": 16, "hx_dout": 34, "hx_sck": 5,
        "reed_front": 35, "reed_shutter": 39, "uv": 32, "halogen": 33, "cob": 2, "fan": 15,
        "rake": [4, 13, 14, 18, 19, 21, 22, 23]}
RAKE_CHANNELS = ["rake%d" % i for i in range(1, 9)]
CHANNELS = RAKE_CHANNELS + ["uv", "halogen", "cob", "fan"]
CHANNEL_KIND = {**{c: "rake" for c in RAKE_CHANNELS}, "uv": "uv", "halogen": "halogen", "cob": "cob", "fan": "fan"}
FIRMWARE_DEFAULTS = {
    "max_on_ms": {"rake": 20000, "uv": 30000, "halogen": 45000, "cob": 900000, "fan": 0},
    "cool_factor": {"rake": 2.0, "uv": 2.0, "halogen": 3.0, "cob": 0.5, "fan": 0.0},
    "hb_timeout_ms": 3000, "dps": 30.0, "accel": 45.0, "max_dps": 90.0, "max_accel": 180.0,
    "motor_steps": 200, "microsteps": 16, "ratio": 14.0, "run_ma": 1000, "hold_pct": 30,
    "door_stops_motor": True, "allow_no_uart": False,
}
HARD_MAX_ON_MS = {"rake": 30000, "uv": 60000, "halogen": 60000, "cob": 1800000, "fan": 0}
HARD_MIN_COOL = {"rake": 1.0, "uv": 1.0, "halogen": 2.0, "cob": 0.0, "fan": 0.0}
MOTOR_STEPS = env("BOX_MOTOR_STEPS", 200)
MICROSTEPS = env("BOX_MICROSTEPS", 16)
GEAR_RATIO = env("BOX_GEAR_RATIO", 14.0)          # 280T printed crown / 20T GT2 pulley
STEPS_PER_REV = int(round(MOTOR_STEPS * MICROSTEPS * GEAR_RATIO))   # 44 800

# --- Scale -----------------------------------------------------------------------------
WEIGH_SAMPLES = env("BOX_WEIGH_SAMPLES", 20)      # stability window (HX711 at 10 samples/s)
WEIGH_STABLE_G = env("BOX_WEIGH_STABLE_G", 0.5)
WEIGH_TIMEOUT_MS = env("BOX_WEIGH_TIMEOUT_MS", 8000)
MAX_OBJECT_G = env("BOX_MAX_OBJECT_G", 3000.0)
TARE_ESTIMATE_G = 1150.0                          # platter + mechanism on the 5 kg cell (CAD R1)

# --- Timings (seconds) -------------------------------------------------------------------
SETTLE_AFTER_ROTATE_S = env("BOX_SETTLE_AFTER_ROTATE_S", 0.6)   # vibration of the platter + object
SETTLE_BEFORE_WEIGH_S = env("BOX_SETTLE_BEFORE_WEIGH_S", 1.5)   # after the door closed / motor off
LIGHT_SETTLE_S = env("BOX_LIGHT_SETTLE_S", 0.15)
THERMAL_BASELINE_S = env("BOX_THERMAL_BASELINE_S", 5.0)
THERMAL_HEAT_S = env("BOX_THERMAL_HEAT_S", 15.0)
THERMAL_COOL_S = env("BOX_THERMAL_COOL_S", 90.0)                # 60..120 s
# Closed-loop heating (yq/box/heatguard.py): the halogen goes off early when the
# object's surface rises THERMAL_MAX_DT_C over the baseline or reaches THERMAL_MAX_ABS_C.
THERMAL_HEAT_ENABLED = env("BOX_THERMAL_HEAT_ENABLED", True)     # False = never heat (no thermography)
THERMAL_MAX_DT_C = env("BOX_THERMAL_MAX_DT_C", 5.0)
THERMAL_MAX_ABS_C = env("BOX_THERMAL_MAX_ABS_C", 35.0)
THERMAL_GUARD_ROI = (0.15, 0.85, 0.10, 0.95)                    # x0, x1, y0, y1 fractions of the frame
THERMAL_WARM_PX_C = env("BOX_THERMAL_WARM_PX_C", 0.4)           # a pixel "warmed" above this rise
THERMAL_MIN_WARM_PX = env("BOX_THERMAL_MIN_WARM_PX", 20)
THERMAL_GUARD_PERCENTILE = env("BOX_THERMAL_GUARD_PERCENTILE", 99.0)
THERMAL_MAX_FROZEN_S = env("BOX_THERMAL_MAX_FROZEN_S", 1.5)     # blind longer than this while heating -> stop
FAN_AFTER_SCAN_S = env("BOX_FAN_AFTER_SCAN_S", 120.0)
# Simulator + mock cameras only: >1 runs the whole scan faster than real time
# (tests use 50). Timestamps written to the scan stay in "virtual" seconds.
SIM_TIME_SCALE = env("BOX_SIM_TIME_SCALE", 1.0)
SIM_OBJECT_G = env("BOX_SIM_OBJECT_G", 812.5)     # synthetic vessel on the simulated platter
MOCK_RESOLUTION = (env("BOX_MOCK_WIDTH", 1600), env("BOX_MOCK_HEIGHT", 1200))

# --- Profiles (CONTRACTS.md §4: turntable stops per camera) -----------------------------
PROFILES = {
    "quick": {"stops": 18, "rti": True, "uv": True, "thermal_cool_s": 60.0},
    "standard": {"stops": 36, "rti": True, "uv": True, "thermal_cool_s": 90.0},
    "detailed": {"stops": 72, "rti": True, "uv": True, "thermal_cool_s": 120.0},
}
PACKAGE_LONG_SIDE_PX = env("BOX_PACKAGE_LONG_SIDE_PX", 3000)    # photogrammetry downscale before upload
PACKAGE_JPEG_QUALITY = env("BOX_PACKAGE_JPEG_QUALITY", 92)

# --- Cameras (2x Arducam IMX519 16 MP USB 3.0 UVC) ---------------------------------------
# Both cameras report the same USB name, so /dev/v4l/by-id may not tell them
# apart. Set YQ_BOX_CAMERA_A / _B to a /dev/v4l/by-path/... or by-id link once
# the cameras are plugged into their final ports (docs/box.md explains how).
CAMERA_A_DEVICE = env("BOX_CAMERA_A", "")
CAMERA_B_DEVICE = env("BOX_CAMERA_B", "")
CAMERA_NAME_HINTS = env("BOX_CAMERA_HINTS", ["Arducam", "IMX519", "16MP"])
CAMERA_RESOLUTION = (env("BOX_CAMERA_WIDTH", 4656), env("BOX_CAMERA_HEIGHT", 3496))
CAMERA_FOURCC = env("BOX_CAMERA_FOURCC", "MJPG")
CAMERA_WARMUP_FRAMES = env("BOX_CAMERA_WARMUP_FRAMES", 5)
CAMERA_FLUSH_FRAMES = env("BOX_CAMERA_FLUSH_FRAMES", 3)          # stale buffered frames after a move
CAMERA_RETRIES = env("BOX_CAMERA_RETRIES", 2)
CAMERA_MIN_SHARPNESS = env("BOX_CAMERA_MIN_SHARPNESS", 20.0)     # variance of Laplacian on 1000 px image
CAMERA_JPEG_QUALITY = env("BOX_CAMERA_JPEG_QUALITY", 95)
# Locked controls per camera. UVC exposure_time_absolute is in 100 us units
# (Arducam UVC docs); the IMX519 USB bridge caps it at 2000 = 200 ms.
# focus: UVC focus_absolute value; measure it once (python -m yq.box.cli focus-sweep A)
# and set YQ_BOX_FOCUS_A / _B. Ranges are read from the device at runtime and clamped.
CAMERA_CONTROLS = {
    "A": {"focus": env("BOX_FOCUS_A", 300), "exposure_us": env("BOX_EXPOSURE_A_US", 20000),
          "gain": env("BOX_GAIN_A", 0), "wb_k": env("BOX_WB_A_K", 4600)},
    "B": {"focus": env("BOX_FOCUS_B", 300), "exposure_us": env("BOX_EXPOSURE_B_US", 20000),
          "gain": env("BOX_GAIN_B", 0), "wb_k": env("BOX_WB_B_K", 4600)},
}
RTI_CAMERA = env("BOX_RTI_CAMERA", "B")          # the steepest view (55 deg) is the best for RTI
RTI_EXPOSURE_US = env("BOX_RTI_EXPOSURE_US", 30000)
UV_CAMERA = env("BOX_UV_CAMERA", "B")
UV_EXPOSURE_US = env("BOX_UV_EXPOSURE_US", 200000)                # UVC maximum of this camera
UV_GAIN = env("BOX_UV_GAIN", 8)
VISIBLE_EXPOSURE_US = env("BOX_VISIBLE_EXPOSURE_US", 20000)
UV_FILTER = env("BOX_UV_FILTER", "ZWB2")         # "" if the visible-cut filter is not mounted

# --- Thermal (GroupGets PureThermal 3 + FLIR Lepton 3.5) -----------------------------------
THERMAL_DEVICE = env("BOX_THERMAL_DEVICE", "")   # "" = /dev/v4l/by-id/*PureThermal*-video-index0
THERMAL_NAME_HINTS = env("BOX_THERMAL_HINTS", ["PureThermal", "GroupGets"])
THERMAL_SIZE = (160, 120)                        # Lepton 3.5, width x height
THERMAL_FPS = 8.7                                # export-limited Lepton frame rate
THERMAL_FROZEN_RUN = env("BOX_THERMAL_FROZEN_RUN", 2)  # identical consecutive frames => FFC freeze

# --- Geometry (object frame, mm) ------------------------------------------------------------
CAD_TO_OBJECT = (150.0, 150.0, 82.0)
PLATTER_RADIUS_MM = 90.0
OBJECT_MAX = {"diameter_mm": 150.0, "height_mm": 150.0, "mass_g": 3000.0}
OBJECT_AIM_MM = (0.0, 0.0, 75.0)                  # optical axes cross at mid object height


def _obj(p):
    return tuple(round(a - b, 2) for a, b in zip(p, CAD_TO_OBJECT))


CAMERA_GEOMETRY = {   # CAD parametros.json "cameras": optical centre, elevation, FOV used by the CAD check
    "A": {"position_mm": _obj((150.0, 410.0, 251.632)), "aim_mm": OBJECT_AIM_MM, "elevation_deg": 20.0,
          "hfov_deg": 65.0, "vfov_deg": 50.0},
    "B": {"position_mm": _obj((150.0, 305.0, 378.363)), "aim_mm": OBJECT_AIM_MM, "elevation_deg": 55.0,
          "hfov_deg": 65.0, "vfov_deg": 50.0},
    "T": {"position_mm": _obj((-135.0, 150.0, 157.0)), "aim_mm": OBJECT_AIM_MM, "elevation_deg": 0.0,
          "hfov_deg": 57.0, "vfov_deg": 43.0},
}
# 8 raking LEDs: centres of LED1..LED8 in Tablas/componentes.json, all aimed at the platter centre.
RAKE_POSITIONS_CAD = [(74.55, 280.8, 150.4), (270.65, 280.65, 150.4), (21.3, 39.4, 150.35),
                      (21.2, 79.55, 150.4), (21.3, 260.6, 150.35), (278.75, 54.45, 150.4),
                      (278.9, 150.0, 150.5), (278.75, 245.55, 150.4)]
UV_POSITION_CAD = (250.0, 250.0, 223.4)
HALOGEN_POSITION_CAD = (246.0, 248.0, 292.0)
COB_HEIGHT_CAD = 345.0


def light_geometry() -> list:
    """lights.json entries: index, channel, position_mm, direction (unit vector platter centre -> LED)."""
    out = []
    for i, p in enumerate(RAKE_POSITIONS_CAD, start=1):
        pos = _obj(p)
        n = math.sqrt(sum(c * c for c in pos))
        out.append({"index": i, "channel": "rake%d" % i, "position_mm": list(pos),
                    "direction": [round(c / n, 5) for c in pos],
                    "elevation_deg": round(math.degrees(math.asin(pos[2] / n)), 2),
                    "azimuth_deg": round(math.degrees(math.atan2(pos[1], pos[0])), 2)})
    return out


def calib_dir() -> Path:
    """Box calibration files live in CALIB_DIR/box (resolved at call time: tests patch CALIB_DIR)."""
    return Path(config.CALIB_DIR) / "box"
