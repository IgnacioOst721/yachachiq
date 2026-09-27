"""Camera controls (v4l2-ctl command generation), image checks and the mock camera."""
import numpy as np
import pytest

from yq.box import uvcctl
from yq.box.cameras import check_image, image_stats

# `v4l2-ctl --list-ctrls-menus` as printed by v4l-utils 1.22 (Ubuntu 22.04 / JetPack 6) for a UVC camera
NEW_KERNEL = """
User Controls

                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
                           gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=0
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=1 (50 Hz)
\t\t\t\t0: Disabled
\t\t\t\t1: 50 Hz
\t\t\t\t2: 60 Hz
      white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=10 default=4600 value=4600 flags=inactive

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
\t\t\t\t1: Manual Mode
\t\t\t\t3: Aperture Priority Mode
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=2000 step=1 default=157 value=157 flags=inactive
     exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
                 focus_absolute 0x009a090a (int)    : min=0 max=1023 step=1 default=0 value=0 flags=inactive
     focus_automatic_continuous 0x009a090c (bool)   : default=0 value=1
"""
# older kernels / v4l-utils name the same controls differently
OLD_KERNEL = """
                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
 white_balance_temperature_auto 0x0098090c (bool)   : default=1 value=1
                           gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=0
      white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=1 default=4600 value=4600 flags=inactive
                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
                exposure_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=157 flags=inactive
                     focus_auto 0x009a090c (bool)   : default=1 value=1
                 focus_absolute 0x009a090a (int)    : min=0 max=1023 step=1 default=0 value=0 flags=inactive
"""


def test_parse_ctrls():
    c = uvcctl.parse_ctrls(NEW_KERNEL)
    assert c["exposure_time_absolute"]["max"] == 2000 and c["exposure_time_absolute"]["flags"] == "inactive"
    assert c["auto_exposure"]["menu"] == {1: "Manual Mode", 3: "Aperture Priority Mode"}
    assert c["focus_automatic_continuous"]["type"] == "bool"


def test_lock_plan_new_names_order_units_and_clamping():
    ctrls = uvcctl.parse_ctrls(NEW_KERNEL)
    stages, applied, warnings = uvcctl.lock_plan(ctrls, focus=300, exposure_us=350000, gain=4, wb_k=4633)
    assert stages[0] == {"auto_exposure": 1, "focus_automatic_continuous": 0, "white_balance_automatic": 0,
                         "exposure_dynamic_framerate": 0}
    # 350 ms asked -> 3500 x 100 us, clamped to the camera's 2000 (200 ms)
    assert stages[1] == {"exposure_time_absolute": 2000, "focus_absolute": 300, "gain": 4,
                         "white_balance_temperature": 4630}
    assert applied["exposure"] == 200000 and any("fuera de rango" in w for w in warnings)
    argv = uvcctl.set_args("/dev/v4l/by-path/x", stages[0])
    assert argv[:4] == ["v4l2-ctl", "-d", "/dev/v4l/by-path/x", "-c"]
    assert argv[4] == "auto_exposure=1,focus_automatic_continuous=0,white_balance_automatic=0,exposure_dynamic_framerate=0"


def test_lock_plan_old_names():
    ctrls = uvcctl.parse_ctrls(OLD_KERNEL)
    stages, applied, _ = uvcctl.lock_plan(ctrls, focus=512, exposure_us=20000, gain=0, wb_k=5000)
    assert stages[0] == {"exposure_auto": 1, "focus_auto": 0, "white_balance_temperature_auto": 0}
    assert stages[1]["exposure_absolute"] == 200 and stages[1]["focus_absolute"] == 512
    assert uvcctl.parse_get("focus_absolute: 512\nexposure_absolute: 200\n") == {"focus_absolute": 512,
                                                                                "exposure_absolute": 200}


def test_image_checks():
    rng = np.random.default_rng(0)
    black = np.full((480, 640, 3), 3, np.uint8)
    assert check_image(image_stats(black), "dark", 20) == []
    assert check_image(image_stats(black), "normal", 20)                 # "casi negra"
    tex = np.zeros((480, 640, 3), np.uint8)
    tex[100:380, 200:440] = rng.integers(60, 200, (280, 240, 3), dtype=np.uint8)
    st = image_stats(tex)
    assert st["sharpness"] > 100 and check_image(st, "normal", 20) == []
    import cv2
    blurred = cv2.GaussianBlur(tex, (0, 0), 8)
    assert any("borrosa" in p for p in check_image(image_stats(blurred), "normal", 20))
    white = np.full((480, 640, 3), 255, np.uint8)
    assert any("sobreexpuesta" in p for p in check_image(image_stats(white), "normal", 0))


def test_mock_camera_follows_platter_and_lights():
    from yq.box.mockcam import render
    scene = {"platter_deg": 0.0, "lights": {"cob": 1.0}, "object_g": 800.0}
    a0 = render("A", scene, (320, 240)).astype(float)
    a0b = render("A", scene, (320, 240), seed=5).astype(float)
    a90 = render("A", {**scene, "platter_deg": 90.0}, (320, 240)).astype(float)
    lit = (a0.max(axis=2) > 30) | (a90.max(axis=2) > 30)   # object + platter markers
    noise = np.abs(a0 - a0b)[lit].mean()
    moved = np.abs(a0 - a90)[lit].mean()
    assert noise < 2.0                                    # same pose: only sensor noise differs
    assert moved > 5 * noise                              # rotated: figure and markers moved
    dark = render("A", {**scene, "lights": {}}, (320, 240))
    assert dark.mean() < 5
    rake = render("B", {**scene, "lights": {"rake7": 1.0}}, (320, 240)).astype(float)
    left, right = rake[:, :160].mean(), rake[:, 160:].mean()
    assert abs(left - right) > 1.0                        # raking light from one side only
    empty = render("A", {**scene, "object_g": 0.0}, (320, 240))
    assert image_stats(empty)["lit_fraction"] < image_stats(a0.astype(np.uint8))["lit_fraction"]


def test_mock_camera_focus_changes_sharpness():
    from yq.box.mockcam import render
    scene = {"platter_deg": 0.0, "lights": {"cob": 1.0}, "object_g": 800.0}
    sharp = image_stats(render("A", scene, (640, 480), focus=300))["sharpness"]
    soft = image_stats(render("A", scene, (640, 480), focus=600))["sharpness"]
    assert sharp > 3 * soft


def test_find_box_cameras_env_override(monkeypatch):
    from yq.box import cameras
    from yq.box import settings as S
    monkeypatch.setattr(S, "CAMERA_A_DEVICE", "/dev/v4l/by-path/usb-0:1.1-video-index0")
    monkeypatch.setattr(S, "CAMERA_B_DEVICE", "/dev/v4l/by-path/usb-0:1.2-video-index0")
    assert cameras.find_box_cameras() == {"A": "/dev/v4l/by-path/usb-0:1.1-video-index0",
                                          "B": "/dev/v4l/by-path/usb-0:1.2-video-index0"}
    monkeypatch.setattr(S, "CAMERA_B_DEVICE", "")
    monkeypatch.setattr(cameras, "list_capture_devices", lambda hints: [])
    with pytest.raises(RuntimeError):
        cameras.find_box_cameras()
