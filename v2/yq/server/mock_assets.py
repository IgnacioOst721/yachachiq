"""Synthetic files for the UI mocks: a 3D vessel (GLB), RTI/PTM relighting data,
UV and thermal images, object photos, camera frames and a pen drawing.

Everything is generated with numpy + Pillow (cv2 is not needed), deterministic,
so the kiosk shows real 3D / relighting / thermal viewers in YQ_MOCK=1 mode on
any computer. None of this is presented as a real measurement: the scan mock
marks its result engine as "mock" and the UI labels it SIMULADO.
"""
from __future__ import annotations

import io
import json
import math
import struct
from pathlib import Path
from typing import Optional

_cache: dict = {}


# --------------------------------------------------------------------------------------
# 3D: a Moche-style stirrup-spout vessel as a binary glTF (GLB)
# --------------------------------------------------------------------------------------
def _lathe(profile, segments, color_fn):
    import numpy as np
    pos, col, idx = [], [], []
    n = len(profile)
    for s in range(segments + 1):
        th = 2 * math.pi * s / segments
        for r, z in profile:
            x, y = r * math.cos(th), r * math.sin(th)
            pos.append((x, z, y))                     # glTF is Y-up
            col.append(color_fn(th, z, r))
    for s in range(segments):
        for i in range(n - 1):
            a = s * n + i
            b = (s + 1) * n + i
            idx += [a, a + 1, b, b, a + 1, b + 1]
    return np.array(pos, "f4"), np.array(col, "f4"), np.array(idx, "u4")


def _tube(path_pts, radius, segments, color):
    """Tube along a 3D polyline (Y-up coordinates)."""
    import numpy as np
    pos, col, idx = [], [], []
    pts = [np.array(p, "f4") for p in path_pts]
    n = len(pts)
    for i, p in enumerate(pts):
        t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        t = t / (np.linalg.norm(t) + 1e-9)
        ref = np.array([0, 0, 1], "f4") if abs(t[2]) < 0.9 else np.array([1, 0, 0], "f4")
        u = np.cross(t, ref)
        u /= np.linalg.norm(u) + 1e-9
        v = np.cross(t, u)
        for s in range(segments + 1):
            a = 2 * math.pi * s / segments
            pos.append(tuple(p + radius * (math.cos(a) * u + math.sin(a) * v)))
            col.append(color)
    for i in range(n - 1):
        for s in range(segments):
            a = i * (segments + 1) + s
            b = (i + 1) * (segments + 1) + s
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return np.array(pos, "f4"), np.array(col, "f4"), np.array(idx, "u4")


def _normals(pos, idx):
    import numpy as np
    nrm = np.zeros_like(pos)
    tri = idx.reshape(-1, 3)
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross((b - a).astype("f8"), (c - a).astype("f8"))
    for k in range(3):
        np.add.at(nrm, tri[:, k], fn)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return (nrm / ln).astype("f4")


def vessel_glb() -> bytes:
    """~140 mm tall stirrup-spout vessel, cream band with a stepped (tocapu) pattern."""
    if "glb" in _cache:
        return _cache["glb"]
    import numpy as np
    terracotta = (0.66, 0.33, 0.20)
    cream = (0.93, 0.84, 0.66)
    dark = (0.42, 0.13, 0.10)

    def body_color(th, z, r):
        if 32 <= z <= 62:
            cell = int(th / (2 * math.pi) * 16) + int((z - 32) / 7.5)
            return dark if cell % 2 == 0 else cream
        if 62 < z <= 66 or 28 <= z < 32:
            return dark
        return terracotta

    body = [(0, 0), (28, 0), (40, 5), (47, 18), (50, 38), (48, 60), (40, 76), (26, 87), (10, 91), (0, 92)]
    parts = [_lathe(body, 64, body_color)]
    # stirrup handle: half torus in the XZ plane (Y-up space: x, y=height)
    arc = [(26 * math.cos(a), 78 + 26 * math.sin(a), 0.0) for a in np.linspace(0, math.pi, 40)]
    parts.append(_tube(arc, 6.5, 20, terracotta))
    # spout from the top of the stirrup, with a flared lip
    spout = [(0.0, 100 + 1.6 * i, 0.0) for i in range(22)]
    parts.append(_tube(spout, 7.0, 24, terracotta))
    lip = [(0, 132), (8.8, 133), (9.5, 136), (7.5, 137), (6, 134), (0, 133)]
    parts.append(_lathe(lip, 24, lambda th, z, r: dark))

    positions, colors, indices, off = [], [], [], 0
    for p, c, i in parts:
        positions.append(p)
        colors.append(c)
        indices.append(i + off)
        off += len(p)
    pos = np.concatenate(positions) / 1000.0          # millimetres -> metres (glTF unit)
    col = np.concatenate(colors) ** 2.2                  # glTF COLOR_0 is linear, the palette above is sRGB
    idx = np.concatenate(indices).astype("u4")
    nrm = _normals(pos, idx)
    _cache["glb"] = write_glb(pos.astype("f4"), nrm, col.astype("f4"), idx)
    return _cache["glb"]


def write_glb(pos, nrm, col, idx) -> bytes:
    """Minimal glTF 2.0 binary with one mesh: POSITION, NORMAL, COLOR_0, indices."""
    blobs = [pos.tobytes(), nrm.tobytes(), col.tobytes(), idx.tobytes()]
    views, offset, bin_ = [], 0, b""
    for i, b in enumerate(blobs):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(b),
                      "target": 34963 if i == 3 else 34962})
        bin_ += b
        offset += len(b)
        pad = (-offset) % 4
        bin_ += b"\0" * pad
        offset += pad
    n = len(pos)
    gltf = {
        "asset": {"version": "2.0", "generator": "yachachiq-ui-mock"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "vessel"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                                    "indices": 3, "material": 0}]}],
        "materials": [{"name": "clay", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                                                "metallicFactor": 0.0, "roughnessFactor": 0.8}}],
        "buffers": [{"byteLength": len(bin_)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
             "min": [float(v) for v in pos.min(0)], "max": [float(v) for v in pos.max(0)]},
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 3, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(bin_), 0x004E4942) + bin_
    return out


# --------------------------------------------------------------------------------------
# RTI: Polynomial Texture Map (LRGB PTM, 6 coefficients) of an incised clay surface
# --------------------------------------------------------------------------------------
PTM_FORMAT = "yq-ptm-lrgb-v1"


def _blur(a, k: int):
    import numpy as np
    if k <= 1:
        return a
    ker = np.exp(-0.5 * (np.arange(-2 * k, 2 * k + 1) / k) ** 2)
    ker /= ker.sum()
    pad = 2 * k
    b = np.pad(a, pad, mode="edge")
    b = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, b)
    b = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 0, b)
    return b[pad:-pad, pad:-pad]


def _relief(size: int):
    """Height map with a stepped cross (chakana), a spiral groove and fine texture."""
    import numpy as np
    yy, xx = np.mgrid[0:size, 0:size].astype("f4") / size - 0.5
    h = np.zeros((size, size), "f4")
    # chakana: union of 3 centred rectangles of growing width / shrinking height
    for w, hh in ((0.14, 0.42), (0.28, 0.28), (0.42, 0.14)):
        h = np.maximum(h, ((abs(xx) < w / 2) & (abs(yy) < hh / 2)).astype("f4"))
    h -= 0.8 * (np.hypot(xx, yy) < 0.045)            # central hole
    r = np.hypot(xx, yy)
    th = np.arctan2(yy, xx)
    spiral = np.abs(((r * 26 - th / (2 * math.pi) * 1.0) % 1.0) - 0.5) < 0.09
    h -= 0.35 * (spiral & (r > 0.3) & (r < 0.47))
    rng = np.random.default_rng(7)
    h += 0.04 * _blur(rng.standard_normal((size, size)).astype("f4"), 2)
    return _blur(h, 3)


def _albedo(size: int):
    import numpy as np
    yy, xx = np.mgrid[0:size, 0:size].astype("f4") / size - 0.5
    base = np.array([0.78, 0.52, 0.36], "f4")
    img = np.ones((size, size, 3), "f4") * base
    band = (np.abs(yy) > 0.36) & (np.abs(yy) < 0.44)
    img[band] = [0.92, 0.84, 0.68]
    paint = ((xx * 10).astype(int) + (yy * 10).astype(int)) % 2 == 0
    img[band & paint] = [0.45, 0.14, 0.11]
    rng = np.random.default_rng(3)
    img *= (0.93 + 0.07 * _blur(rng.random((size, size)).astype("f4"), 4))[..., None]
    return np.clip(img, 0, 1)


def ptm_files(out_dir: Path, size: int = 384) -> dict:
    """Write ptm.json + 2 coefficient PNGs + albedo PNG into out_dir (see docs/ui.md)."""
    import numpy as np
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = _relief(size) * 9.0
    # u to the right, v up (towards the top of the image): rows grow downwards
    dhdu = np.gradient(h, axis=1)
    dhdv = -np.gradient(h, axis=0)
    n = np.stack([-dhdu, -dhdv, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    lights = []
    for elev in (18, 35, 55, 75):
        for k in range(12):
            az = 2 * math.pi * k / 12 + elev
            e = math.radians(elev)
            lights.append((math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)))
    L = np.array(lights, "f8")                                  # J x 3
    with np.errstate(all="ignore"):                             # Accelerate BLAS raises spurious FP flags
        lum = np.clip(n.reshape(-1, 3).astype("f8") @ L.T, 0, None)   # P x J
        lu, lv = L[:, 0], L[:, 1]
        A = np.stack([lu * lu, lv * lv, lu * lv, lu, lv, np.ones_like(lu)], 1)   # J x 6
        coef = (np.linalg.pinv(A) @ lum.T).T                    # P x 6
    coef = np.nan_to_num(coef)
    scale, bias, q = [], [], []
    for i in range(6):
        c = coef[:, i]
        lo, hi = float(np.percentile(c, 0.2)), float(np.percentile(c, 99.8))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        scale.append(hi - lo)
        bias.append(lo)
        q.append(np.clip(np.round((c - lo) / (hi - lo) * 255), 0, 255).astype("u1").reshape(size, size))
    Image.fromarray(np.stack(q[0:3], -1), "RGB").save(out_dir / "ptm_coeffs_0.png")
    Image.fromarray(np.stack(q[3:6], -1), "RGB").save(out_dir / "ptm_coeffs_1.png")
    Image.fromarray((_albedo(size) * 255).astype("u1"), "RGB").save(out_dir / "ptm_albedo.png")
    meta = {
        "format": PTM_FORMAT, "width": size, "height": size,
        "model": "L(lu,lv) = a0*lu^2 + a1*lv^2 + a2*lu*lv + a3*lu + a4*lv + a5",
        "coeff_images": ["ptm_coeffs_0.png", "ptm_coeffs_1.png"],
        "channels": [["a0", "a1", "a2"], ["a3", "a4", "a5"]],
        "decode": "a_i = bias[i] + scale[i] * (byte / 255)",
        "scale": [round(s, 6) for s in scale], "bias": [round(b, 6) for b in bias],
        "albedo": "ptm_albedo.png",
        "light_frame": "lu to the image right, lv to the image top, unit light vector (lu, lv, lw)",
        "px_per_mm": 4.0, "lights_used": len(lights), "source": "mock",
    }
    (out_dir / "ptm.json").write_text(json.dumps(meta, indent=1))
    return meta


# --------------------------------------------------------------------------------------
# Images: colour maps, thermal, UV, photos
# --------------------------------------------------------------------------------------
_IRON = [(0.0, (0, 0, 10)), (0.15, (40, 0, 90)), (0.35, (140, 10, 130)), (0.55, (220, 60, 40)),
         (0.75, (250, 150, 0)), (0.9, (255, 220, 60)), (1.0, (255, 255, 230))]


def iron(values):
    """0..1 array -> uint8 RGB with an 'ironbow' thermal palette."""
    import numpy as np
    v = np.clip(values, 0, 1)
    xs = [p for p, _ in _IRON]
    out = np.zeros(v.shape + (3,), "f4")
    for c in range(3):
        out[..., c] = np.interp(v, xs, [col[c] for _, col in _IRON])
    return out.astype("u1")


def _png(arr) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _jpeg(img, q: int = 82) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    return buf.getvalue()


def thermal_field(t: float, w: int = 160, h: int = 120):
    """Temperatures (deg C) of the object while heating (t 0..1) / cooling (t 1..2)."""
    import numpy as np
    yy, xx = np.mgrid[0:h, 0:w].astype("f4")
    cx, cy = w / 2, h * 0.55
    obj = ((xx - cx) / 38) ** 2 + ((yy - cy) / 46) ** 2 < 1
    heat = t if t <= 1 else max(0.0, 1 - (t - 1) * 0.8)
    temp = np.full((h, w), 22.0, "f4")
    temp[obj] += 14 * heat
    # a hidden crack/fill cools slower (the "anomaly")
    blob = ((xx - cx - 12) / 7) ** 2 + ((yy - cy + 8) / 14) ** 2 < 1
    extra = 5 * (heat if t <= 1 else min(1.0, heat + 0.35))
    temp[blob & obj] += extra
    return _blur(temp, 2)


def thermal_png(t: float, lo: float = 20.0, hi: float = 44.0) -> bytes:
    import numpy as np
    temp = thermal_field(t)
    return _png(np.kron(iron((temp - lo) / (hi - lo)), np.ones((3, 3, 1), "u1")))


def object_photo(angle_deg: float = 0.0, light: str = "white", w: int = 480, h: int = 360) -> bytes:
    """A side view of the vessel on the turntable (JPEG)."""
    from PIL import Image, ImageDraw
    bg = {"white": (228, 222, 210), "uv": (16, 6, 40), "dark": (6, 6, 8)}.get(light, (70, 62, 58))
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    cx, base = w // 2, int(h * 0.88)
    s = h / 175.0
    body = [(0, 0), (28, 0), (40, 5), (47, 18), (50, 38), (48, 60), (40, 76), (26, 87), (10, 91)]
    pts = [(cx + r * s, base - z * s) for r, z in body] + [(cx - r * s, base - z * s) for r, z in reversed(body)]
    if light == "uv":
        fill, band = (70, 40, 140), (110, 70, 190)
    elif light.startswith("led"):
        k = int(light[3:] or 1)
        g = 0.55 + 0.45 * math.cos(k * 0.8)
        fill, band = tuple(int(c * g) for c in (168, 84, 52)), tuple(int(c * g) for c in (236, 214, 170))
    else:
        fill, band = (168, 84, 52), (236, 214, 170)
    d.polygon(pts, fill=fill)
    d.rectangle([cx - 49 * s, base - 62 * s, cx + 49 * s, base - 32 * s], fill=band)
    off = (angle_deg / 360.0) * 16
    for i in range(-10, 11):
        x = cx + (i + off % 2) * 6.2 * s
        if abs(x - cx) < 46 * s:
            d.rectangle([x, base - 58 * s, x + 3 * s, base - 36 * s], fill=(110, 34, 26))
    d.arc([cx - 26 * s, base - 104 * s, cx + 26 * s, base - 52 * s], 180, 360, fill=fill, width=int(12 * s))
    d.rectangle([cx - 7 * s, base - 135 * s, cx + 7 * s, base - 102 * s], fill=fill)
    if light == "uv":
        d.ellipse([cx + 6 * s, base - 70 * s, cx + 30 * s, base - 44 * s], fill=(190, 255, 120))
    d.ellipse([cx - 80 * s, base - 6 * s, cx + 80 * s, base + 10 * s], outline=(90, 90, 90), width=2)
    return _jpeg(img)


def uv_overlay(w: int = 480, h: int = 360) -> bytes:
    from PIL import Image, ImageDraw
    img = Image.open(io.BytesIO(object_photo(0, "white", w, h)))
    d = ImageDraw.Draw(img)
    s = h / 175.0
    cx, base = w // 2, int(h * 0.88)
    d.ellipse([cx + 2 * s, base - 74 * s, cx + 34 * s, base - 40 * s], outline=(255, 30, 140), width=5)
    return _jpeg(img, 88)


def thermal_result(kind: str) -> bytes:
    """'max' = hottest frame; 'anomaly' = where the cooling rate differs (highlighted)."""
    import numpy as np
    if kind == "max":
        return thermal_png(1.0)
    a = thermal_field(1.6) - thermal_field(1.0) * 0.55
    a = (a - a.min()) / (np.ptp(a) + 1e-6)
    return _png(np.kron(iron(a ** 2), np.ones((3, 3, 1), "u1")))


# --------------------------------------------------------------------------------------
# Visitor-facing camera frames (portrait/consent) and sign-language preview
# --------------------------------------------------------------------------------------
def portrait_frame(t: float, covered: bool = False, w: int = 640, h: int = 480) -> bytes:
    from PIL import Image, ImageDraw
    if covered:
        img = Image.new("RGB", (w, h), (9, 7, 8))
        return _jpeg(img)
    img = Image.new("RGB", (w, h), (40, 48, 70))
    d = ImageDraw.Draw(img)
    for i in range(0, h, 8):
        c = int(60 + 60 * i / h)
        d.rectangle([0, i, w, i + 8], fill=(c, int(c * 0.8), 110))
    bob = 6 * math.sin(t * 2.0)
    cx, cy = w // 2, int(h * 0.45 + bob)
    d.ellipse([cx - 160, h - 120, cx + 160, h + 200], fill=(200, 40, 70))          # shoulders (poncho)
    d.ellipse([cx - 80, cy - 100, cx + 80, cy + 100], fill=(196, 140, 100))         # face
    d.pieslice([cx - 92, cy - 120, cx + 92, cy - 10], 180, 360, fill=(30, 20, 20))  # hair
    d.ellipse([cx - 40, cy - 20, cx - 22, cy - 2], fill=(20, 20, 20))
    d.ellipse([cx + 22, cy - 20, cx + 40, cy - 2], fill=(20, 20, 20))
    d.arc([cx - 40, cy + 10, cx + 40, cy + 60], 20, 160, fill=(90, 20, 30), width=6)
    return _jpeg(img, 78)


_HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def sign_frame(t: float, letter: str = "", hands: bool = True, w: int = 640, h: int = 480) -> bytes:
    """Person silhouette + 21-point hand skeleton, like a pose-model preview."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), (24, 26, 36))
    d = ImageDraw.Draw(img)
    cx = w // 2
    d.ellipse([cx - 55, 60, cx + 55, 180], fill=(70, 74, 92))
    d.rounded_rectangle([cx - 150, 190, cx + 150, h + 40], 60, fill=(62, 66, 84))
    if hands:
        wx, wy = cx + 110 + 14 * math.sin(t * 1.7), 300 + 10 * math.cos(t * 2.3)
        pts = [(wx, wy)]
        for f in range(5):
            base_a = -2.2 + f * 0.38
            bend = 0.35 + 0.35 * math.sin(t * 3 + f + (ord(letter[0]) if letter else 0))
            x, y, a = wx, wy, base_a
            for seg in range(4):
                ln = (38, 30, 24, 20)[seg] * (0.8 if f == 0 else 1.0)
                a += bend * (0.4 if seg else 0)
                x, y = x + ln * math.cos(a), y + ln * math.sin(a)
                pts.append((x, y))
        for a, b in _HAND:
            d.line([pts[a], pts[b]], fill=(255, 196, 35), width=5)
        for i, p in enumerate(pts):
            r = 7 if i in (4, 8, 12, 16, 20) else 5
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(0, 200, 170) if i else (255, 60, 120))
        d.line([(cx + 60, 200), (wx, wy)], fill=(255, 60, 120), width=6)
    return _jpeg(img, 75)


# --------------------------------------------------------------------------------------
# Pen drawing (front) and story page (back) for the art mock
# --------------------------------------------------------------------------------------
def drawing_polylines(seed: int = 1) -> list:
    """Andean scene as pen strokes in millimetres on an A4 page (210 x 297)."""
    import random
    rnd = random.Random(seed)
    lines = []
    # frame with stepped corners
    m = 12
    lines.append([(m, m), (210 - m, m), (210 - m, 297 - m), (m, 297 - m), (m, m)])
    # mountains
    x, pts = m, []
    while x < 210 - m:
        pts.append((x, 150 + rnd.uniform(-8, 8)))
        x += rnd.uniform(14, 26)
        pts.append((min(x, 210 - m), 70 + rnd.uniform(0, 40)))
        x += rnd.uniform(14, 26)
    pts.append((210 - m, 150))
    lines.append(pts)
    # snow caps
    for i in range(1, len(pts) - 1, 2):
        px, py = pts[i]
        lines.append([(px - 7, py + 9), (px - 2, py + 5), (px + 2, py + 10), (px + 7, py + 8)])
    # sun with rays
    sx, sy, sr = 160, 48, 13
    lines.append([(sx + sr * math.cos(a / 24 * 2 * math.pi), sy + sr * math.sin(a / 24 * 2 * math.pi)) for a in range(25)])
    for k in range(12):
        a = k / 12 * 2 * math.pi
        lines.append([(sx + (sr + 4) * math.cos(a), sy + (sr + 4) * math.sin(a)),
                      (sx + (sr + 11) * math.cos(a), sy + (sr + 11) * math.sin(a))])
    # condor
    cx, cy = 80, 55
    lines.append([(cx - 38, cy + 6), (cx - 22, cy - 4), (cx - 10, cy + 2), (cx, cy - 2),
                  (cx + 10, cy + 2), (cx + 22, cy - 4), (cx + 38, cy + 6)])
    lines.append([(cx - 3, cy - 2), (cx, cy - 7), (cx + 3, cy - 2)])
    # lake waves
    for row in range(6):
        y = 175 + row * 9
        lines.append([(m + 6 + t * 3, y + 2 * math.sin(t * 0.9 + row)) for t in range(int((210 - 2 * m - 12) / 3))])
    # llama
    lx, ly = 60, 250
    lines.append([(lx, ly), (lx + 4, ly - 20), (lx + 30, ly - 20), (lx + 34, ly), (lx + 30, ly - 20),
                  (lx + 36, ly - 44), (lx + 42, ly - 44), (lx + 37, ly - 38)])
    lines.append([(lx + 4, ly - 20), (lx + 2, ly - 22)])
    # stepped chakana motif
    ox, oy, s = 150, 245, 5
    ch = [(0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (2, 2), (2, 3), (1, 3), (1, 2), (0, 2), (0, 1)]
    lines.append([(ox + a * s * 2, oy + b * s * 2) for a, b in ch])
    return lines


def drawing_png(lines: list, w_px: int = 840, h_mm: float = 297.0, w_mm: float = 210.0) -> bytes:
    from PIL import Image, ImageDraw
    k = w_px / w_mm
    img = Image.new("RGB", (w_px, int(h_mm * k)), (255, 252, 244))
    d = ImageDraw.Draw(img)
    for pl in lines:
        d.line([(x * k, y * k) for x, y in pl], fill=(30, 22, 40), width=3, joint="curve")
    return _png_img(img)


def _png_img(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def svg_paths(lines: list, w_mm: float = 210.0, h_mm: float = 297.0, extra: str = "") -> str:
    paths = []
    for pl in lines:
        d = "M" + " L".join("%.2f %.2f" % (x, y) for x, y in pl)
        paths.append('<path d="%s"/>' % d)
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%gmm" height="%gmm" viewBox="0 0 %g %g">'
            '<g id="pen-black" fill="none" stroke="#000" stroke-width="0.4" stroke-linecap="round">%s</g>%s</svg>'
            % (w_mm, h_mm, w_mm, h_mm, "".join(paths), extra))


def back_svg(title: str, text: str, qr_url: str, w_mm: float = 210.0, h_mm: float = 297.0) -> str:
    """Back page: title + story text (as SVG <text>, the real ART module uses single-line fonts)."""
    import textwrap
    from xml.sax.saxutils import escape
    rows = ['<text x="20" y="30" font-size="9" font-family="sans-serif">%s</text>' % escape(title)]
    y = 44
    for para in text.split("\n"):
        for line in textwrap.wrap(para, 52) or [""]:
            rows.append('<text x="20" y="%.1f" font-size="5.2" font-family="sans-serif">%s</text>' % (y, escape(line)))
            y += 7.5
            if y > 230:
                break
    rows.append('<rect x="150" y="235" width="40" height="40" fill="none" stroke="#000"/>'
                '<text x="152" y="258" font-size="4">QR (mock)</text>'
                '<text x="20" y="280" font-size="4" font-family="sans-serif">%s</text>' % escape(qr_url))
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%gmm" height="%gmm" viewBox="0 0 %g %g">'
            '<g id="pen-black" fill="#000">%s</g></svg>' % (w_mm, h_mm, w_mm, h_mm, "".join(rows)))


def wav_tone(seconds: float, rate: int = 16000) -> bytes:
    """Quiet pentatonic melody as a WAV (mock narration for the hologram package)."""
    import wave
    import numpy as np
    n = max(1, int(seconds * rate))
    t = np.arange(n) / rate
    notes = [392.0, 440.0, 523.3, 587.3, 659.3]
    f = np.array([notes[int(x * 3) % 5] for x in t], "f4")
    y = 0.12 * np.sin(2 * math.pi * f * t) * np.minimum(1, (n - np.arange(n)) / 800)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((y * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def write(path: Path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def placeholder_jpeg(text: str = "", w: int = 640, h: int = 480) -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), (30, 20, 50))
    d = ImageDraw.Draw(img)
    d.text((20, h // 2), text, fill=(255, 200, 60))
    return _jpeg(img)


def ptm_ok(meta: Optional[dict]) -> bool:
    return bool(meta) and meta.get("format") == PTM_FORMAT
