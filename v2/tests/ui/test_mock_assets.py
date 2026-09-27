"""The synthetic viewer data is valid: GLB parses, PTM decodes to the right shading."""
from __future__ import annotations

import json
import math
import struct

import numpy as np
from PIL import Image

from yq.server import mock_assets as A


def test_glb_is_valid_gltf2():
    glb = A.vessel_glb()
    magic, version, total = struct.unpack("<III", glb[:12])
    assert (magic, version, total) == (0x46546C67, 2, len(glb))
    jlen, jtype = struct.unpack("<II", glb[12:20])
    assert jtype == 0x4E4F534A and jlen % 4 == 0
    gltf = json.loads(glb[20:20 + jlen])
    blen, btype = struct.unpack("<II", glb[20 + jlen:28 + jlen])
    assert btype == 0x004E4942 and blen == gltf["buffers"][0]["byteLength"]
    pos = gltf["accessors"][0]
    assert pos["type"] == "VEC3" and len(pos["min"]) == 3
    height_m = pos["max"][1] - pos["min"][1]
    assert 0.12 < height_m < 0.16                                  # ~140 mm vessel, glTF metres, Y up
    for v in gltf["bufferViews"]:
        assert v["byteOffset"] % 4 == 0 and v["byteOffset"] + v["byteLength"] <= blen


def _decode(folder, meta, lu, lv):
    c0 = np.asarray(Image.open(folder / meta["coeff_images"][0]), "f4") / 255.0
    c1 = np.asarray(Image.open(folder / meta["coeff_images"][1]), "f4") / 255.0
    a = [meta["bias"][i] + meta["scale"][i] * (c0 if i < 3 else c1)[..., i % 3] for i in range(6)]
    return a[0] * lu * lu + a[1] * lv * lv + a[2] * lu * lv + a[3] * lu + a[4] * lv + a[5]


def test_ptm_decodes_to_lambertian_shading(tmp_path):
    meta = A.ptm_files(tmp_path, size=128)
    assert meta["format"] == A.PTM_FORMAT and A.ptm_ok(meta)
    h = A._relief(128) * 9.0
    n = np.stack([-np.gradient(h, axis=1), np.gradient(h, axis=0), np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    for az in (0, 90, 200):
        e = math.radians(40)
        l = np.array([math.cos(e) * math.cos(math.radians(az)), math.cos(e) * math.sin(math.radians(az)), math.sin(e)])
        truth = np.clip(n @ l, 0, None)
        got = _decode(tmp_path, meta, l[0], l[1])
        assert float(np.mean(np.abs(got - truth))) < 0.05, az
    # light from the right makes right-facing slopes brighter than left-facing ones
    right = _decode(tmp_path, meta, 0.9, 0.0)
    left = _decode(tmp_path, meta, -0.9, 0.0)
    facing_right = n[..., 0] > 0.2
    assert right[facing_right].mean() > left[facing_right].mean()


def test_images_and_frames():
    for data in (A.thermal_png(0.5), A.thermal_result("anomaly"), A.drawing_png(A.drawing_polylines())):
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
    for data in (A.object_photo(90, "uv"), A.portrait_frame(1.0), A.sign_frame(1.0, "a"), A.uv_overlay()):
        assert data[:2] == b"\xff\xd8"
    svg = A.svg_paths(A.drawing_polylines())
    assert 'width="210mm"' in svg and svg.count("<path") > 20
    assert A.wav_tone(0.3)[:4] == b"RIFF"
