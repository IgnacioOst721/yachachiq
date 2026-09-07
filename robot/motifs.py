"""Procedural Andean motifs drawn as pen strokes (no image model needed).

Every motif is a list of polylines in millimetres, y grows UP (paper space).
    all_keys()                              -> list of motif names
    compose(elements, W, H, seed=None)      -> polylines for a whole scene
    stepped_border(W, H, inset)             -> decorative Andean border

Used as the offline fallback when the laptop is missing, and always for the
border that frames the AI-generated drawings.
"""
import math
import random


# --- primitives -------------------------------------------------------------------

def _circle(cx, cy, r, n=40, a0=0.0, a1=2 * math.pi, close=True):
    pts = [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
    return pts if (close or not close) else pts


def _poly(*pts):
    return [tuple(p) for p in pts]


def _shift(pls, dx, dy):
    return [[(x + dx, y + dy) for x, y in pl] for pl in pls]


def _scale(pls, s, cx=0.0, cy=0.0):
    return [[(cx + (x - cx) * s, cy + (y - cy) * s) for x, y in pl] for pl in pls]


def _mirror(pls, cx=0.0):
    return [[(2 * cx - x, y) for x, y in pl] for pl in pls]


def _wave(x0, x1, y, amp, waves, n=60):
    return [(x0 + (x1 - x0) * i / n, y + amp * math.sin(2 * math.pi * waves * i / n)) for i in range(n + 1)]


def _zigzag(x0, x1, y, amp, steps):
    pts = []
    for i in range(steps + 1):
        x = x0 + (x1 - x0) * i / steps
        pts.append((x, y + (amp if i % 2 else -amp)))
    return pts


# --- motifs (unit size ~ 1.0, centred at 0,0; scaled later) -------------------------

def m_sun():
    pls = [_circle(0, 0, 0.32)]
    for i in range(12):
        a = 2 * math.pi * i / 12
        pls.append(_poly((0.42 * math.cos(a), 0.42 * math.sin(a)), (0.6 * math.cos(a), 0.6 * math.sin(a))))
    # inti face
    pls.append(_poly((-0.12, 0.06), (-0.06, 0.1), (0.0, 0.06)))
    pls.append(_poly((0.0, 0.06), (0.06, 0.1), (0.12, 0.06)))
    pls.append(_circle(0, -0.1, 0.1, n=16, a0=math.pi, a1=2 * math.pi))
    return pls


def m_moon():
    outer = _circle(0, 0, 0.45, n=40, a0=math.pi / 2 - 0.2, a1=3 * math.pi / 2 + 0.2)
    inner = _circle(0.18, 0, 0.36, n=40, a0=math.pi / 2 + 0.1, a1=3 * math.pi / 2 - 0.1)
    return [outer, inner, _poly((-0.1, 0.1), (-0.05, 0.14)), _poly((-0.1, -0.05), (-0.05, -0.01)),
            _circle(0.7, 0.5, 0.05, n=8)]


def m_star():
    pts = []
    for i in range(11):
        r = 0.5 if i % 2 == 0 else 0.2
        a = math.pi / 2 + 2 * math.pi * i / 10
        pts.append((r * math.cos(a), r * math.sin(a)))
    return [pts, _circle(0, 0, 0.08, n=10), _poly((-0.7, 0.3), (-0.6, 0.4)), _poly((0.6, -0.3), (0.7, -0.2))]


def m_cloud():
    pls = [_circle(-0.3, 0, 0.22, n=20, a0=math.pi / 2, a1=3 * math.pi / 2),
           _circle(0, 0.12, 0.3, n=24, a0=0, a1=math.pi),
           _circle(0.32, 0, 0.22, n=20, a0=-math.pi / 2, a1=math.pi / 2),
           _poly((-0.3, -0.22), (0.32, -0.22))]
    return pls


def m_rain():
    pls = m_cloud()
    for i in range(4):
        x = -0.35 + 0.23 * i
        pls.append(_poly((x, -0.3), (x - 0.06, -0.55)))
    return pls


def m_mountain():
    return [_poly((-1.0, -0.5), (-0.4, 0.5), (0.1, -0.1), (0.5, 0.35), (1.0, -0.5)),
            _poly((-0.55, 0.25), (-0.4, 0.5), (-0.25, 0.25), (-0.33, 0.3), (-0.4, 0.22), (-0.47, 0.3)),
            _poly((0.38, 0.2), (0.5, 0.35), (0.62, 0.2), (0.55, 0.24), (0.5, 0.15), (0.45, 0.24)),
            _poly((-0.7, -0.2), (-0.6, -0.05)), _poly((0.7, -0.2), (0.8, -0.05))]


def m_river():
    return [_wave(-1, 1, 0.05, 0.03, 2.5), _wave(-1, 1, -0.05, 0.03, 2.5, n=60),
            _wave(-0.6, -0.2, 0.0, 0.02, 1, n=10), _wave(0.2, 0.6, 0.0, 0.02, 1, n=10)]


def m_lake():
    e = [(0.9 * math.cos(a), 0.35 * math.sin(a)) for a in [2 * math.pi * i / 48 for i in range(49)]]
    return [e, _wave(-0.5, -0.1, 0.05, 0.03, 1, n=12), _wave(0.0, 0.45, -0.05, 0.03, 1, n=12),
            _wave(-0.3, 0.2, -0.15, 0.03, 1, n=12), _poly((-0.95, -0.38), (-0.6, -0.42))]


def m_sea():
    return [_wave(-1, 1, 0.2, 0.08, 3), _wave(-1, 1, 0.0, 0.08, 3), _wave(-1, 1, -0.2, 0.08, 3),
            _poly((-0.9, 0.35), (-0.7, 0.42), (-0.5, 0.35))]


def m_tree():
    return [_poly((-0.05, -0.6), (-0.05, -0.1)), _poly((0.05, -0.6), (0.05, -0.1)),
            _circle(0, 0.15, 0.32, n=30), _circle(-0.2, 0.0, 0.18, n=16, a0=math.pi / 2, a1=3 * math.pi / 2),
            _circle(0.2, 0.0, 0.18, n=16, a0=-math.pi / 2, a1=math.pi / 2), _poly((-0.35, -0.6), (0.35, -0.6))]


def m_flower():
    pls = [_poly((0, -0.6), (0, -0.05)), _circle(0, 0.1, 0.1, n=12)]
    for i in range(6):
        a = 2 * math.pi * i / 6
        pls.append(_circle(0.22 * math.cos(a), 0.1 + 0.22 * math.sin(a), 0.12, n=12))
    pls.append(_poly((0, -0.35), (0.18, -0.45), (0.05, -0.25)))
    return pls


def m_corn():
    pls = [_poly((0, -0.6), (0, 0.5)), _poly((0, -0.2), (-0.35, -0.45), (-0.05, -0.15)),
           _poly((0, 0.0), (0.35, -0.25), (0.05, 0.05)), _poly((0, 0.2), (-0.3, 0.0), (-0.03, 0.25))]
    cob = [(0.18 * math.cos(a) + 0.12, 0.25 + 0.3 * math.sin(a)) for a in [2 * math.pi * i / 24 for i in range(25)]]
    pls.append(cob)
    return pls


def m_potato():
    return [_circle(0, 0, 0.3, n=24), _circle(-0.4, 0.15, 0.22, n=18), _circle(0.4, -0.1, 0.24, n=18),
            _poly((-0.1, 0.05), (-0.05, 0.08)), _poly((0.1, -0.08), (0.14, -0.04))]


def m_field():
    pls = []
    for i in range(4):
        y = -0.4 + 0.25 * i
        pls.append(_poly((-1.0, y), (1.0, y + 0.05)))
    pls.append(_poly((-0.6, -0.4), (-0.55, 0.4)))
    return pls


def m_house():
    return [_poly((-0.5, -0.5), (-0.5, 0.1), (0.5, 0.1), (0.5, -0.5), (-0.5, -0.5)),
            _poly((-0.6, 0.1), (0.0, 0.55), (0.6, 0.1)),
            _poly((-0.12, -0.5), (-0.12, -0.15), (0.12, -0.15), (0.12, -0.5)),
            _poly((0.2, -0.05), (0.2, -0.3), (0.4, -0.3), (0.4, -0.05), (0.2, -0.05)),
            _poly((-0.3, 0.2), (-0.15, 0.32)), _poly((0.1, 0.35), (0.25, 0.46))]


def m_bridge():
    return [_circle(0, -0.5, 0.6, n=30, a0=0, a1=math.pi), _circle(0, -0.5, 0.45, n=24, a0=0, a1=math.pi),
            _poly((-0.9, 0.15), (0.9, 0.15)), _poly((-0.9, 0.3), (0.9, 0.3)),
            _poly((-0.6, 0.3), (-0.6, 0.15)), _poly((0.0, 0.3), (0.0, 0.15)), _poly((0.6, 0.3), (0.6, 0.15))]


def m_path():
    return [_wave(-1, 1, 0.1, 0.05, 1.5), _wave(-1, 1, -0.1, 0.05, 1.5),
            _poly((-0.5, 0.0), (-0.45, 0.0)), _poly((0.0, 0.0), (0.05, 0.0)), _poly((0.5, 0.0), (0.55, 0.0))]


def m_temple():
    pls = [_poly((-0.8, -0.5), (0.8, -0.5)), _poly((-0.6, -0.5), (-0.6, -0.25), (0.6, -0.25), (0.6, -0.5)),
           _poly((-0.45, -0.25), (-0.45, 0.0), (0.45, 0.0), (0.45, -0.25)),
           _poly((-0.3, 0.0), (-0.3, 0.25), (0.3, 0.25), (0.3, 0.0)),
           _poly((-0.35, 0.25), (0.0, 0.5), (0.35, 0.25)),
           _poly((-0.1, -0.5), (-0.1, -0.3), (0.1, -0.3), (0.1, -0.5))]
    return pls


def m_boat():
    return [_poly((-0.7, 0.0), (-0.5, -0.3), (0.5, -0.3), (0.7, 0.0), (-0.7, 0.0)),
            _poly((0, 0), (0, 0.6)), _poly((0, 0.6), (0.4, 0.15), (0, 0.15)),
            _wave(-0.9, 0.9, -0.4, 0.04, 3, n=30), _poly((-0.3, -0.1), (0.3, -0.1))]


def m_fire():
    return [_poly((-0.35, -0.4), (0, 0.6), (0.35, -0.4)), _poly((-0.18, -0.4), (0.05, 0.25), (0.2, -0.4)),
            _poly((-0.6, -0.4), (0.6, -0.4)), _poly((-0.5, -0.5), (-0.3, -0.3)), _poly((0.5, -0.5), (0.3, -0.3))]


def m_music():
    # a quena (Andean flute) with notes
    pls = [_poly((-0.6, -0.4), (0.6, 0.2)), _poly((-0.58, -0.32), (0.62, 0.28)), _poly((-0.6, -0.4), (-0.58, -0.32)),
           _poly((0.6, 0.2), (0.62, 0.28))]
    for i in range(4):
        t = 0.15 + 0.2 * i
        pls.append(_circle(-0.6 + 1.2 * t, -0.4 + 0.6 * t + 0.03, 0.03, n=8))
    pls.append(_circle(0.5, 0.6, 0.08, n=10)); pls.append(_poly((0.58, 0.6), (0.58, 0.95)))
    pls.append(_circle(0.8, 0.45, 0.08, n=10)); pls.append(_poly((0.88, 0.45), (0.88, 0.8)))
    return pls


def m_person():
    return [_circle(0, 0.45, 0.14, n=18),                                   # head
            _poly((-0.18, 0.25), (0.18, 0.25), (0.22, -0.15), (-0.22, -0.15), (-0.18, 0.25)),  # poncho
            _zigzag(-0.2, 0.2, 0.05, 0.03, 8),                                 # poncho stripe
            _poly((-0.18, 0.2), (-0.35, -0.05)), _poly((0.18, 0.2), (0.35, -0.05)),  # arms
            _poly((-0.1, -0.15), (-0.12, -0.6)), _poly((0.1, -0.15), (0.12, -0.6)),  # legs
            _poly((-0.25, 0.55), (0.25, 0.55), (0.18, 0.62), (-0.18, 0.62), (-0.25, 0.55))]  # hat


def m_llama():
    return [_poly((-0.5, -0.1), (-0.5, 0.25), (0.35, 0.25), (0.35, -0.1)),     # body
            _poly((0.35, 0.25), (0.45, 0.7), (0.65, 0.75), (0.7, 0.62), (0.5, 0.55), (0.45, 0.45)),  # neck+head
            _poly((0.55, 0.75), (0.5, 0.88)), _poly((0.62, 0.76), (0.6, 0.9)),  # ears
            _poly((-0.42, -0.1), (-0.42, -0.6)), _poly((-0.25, -0.1), (-0.25, -0.6)),
            _poly((0.1, -0.1), (0.1, -0.6)), _poly((0.27, -0.1), (0.27, -0.6)),
            _poly((-0.5, 0.2), (-0.62, 0.05)),                                 # tail
            _circle(0.62, 0.66, 0.02, n=6)]


def m_condor():
    return [_poly((-1.0, 0.3), (-0.6, 0.45), (-0.2, 0.2), (0.0, 0.05), (0.2, 0.2), (0.6, 0.45), (1.0, 0.3)),
            _poly((-1.0, 0.3), (-0.7, 0.2), (-0.35, 0.05), (0.0, -0.1), (0.35, 0.05), (0.7, 0.2), (1.0, 0.3)),
            _poly((-0.85, 0.28), (-0.9, 0.12)), _poly((-0.75, 0.24), (-0.8, 0.08)),
            _poly((0.85, 0.28), (0.9, 0.12)), _poly((0.75, 0.24), (0.8, 0.08)),
            _circle(0.0, -0.02, 0.09, n=14), _poly((0.06, -0.06), (0.16, -0.12)),
            _poly((-0.08, -0.16), (0.0, -0.3), (0.08, -0.16))]


def m_bird():
    return [_circle(-0.5, 0.2, 0.5, n=16, a0=0.2, a1=1.4), _circle(0.5, 0.2, 0.5, n=16, a0=math.pi - 1.4, a1=math.pi - 0.2),
            _circle(0, 0.05, 0.08, n=10), _poly((0.07, 0.05), (0.15, 0.02))]


def m_fish():
    return [_poly((-0.5, 0), (-0.2, 0.25), (0.3, 0.25), (0.55, 0.0), (0.3, -0.25), (-0.2, -0.25), (-0.5, 0)),
            _poly((-0.5, 0), (-0.75, 0.25), (-0.75, -0.25), (-0.5, 0)),
            _circle(0.35, 0.05, 0.04, n=8), _poly((0.2, 0.25), (0.15, 0.4), (0.0, 0.25)),
            _wave(-0.2, 0.3, 0.0, 0.04, 2, n=12)]


def m_puma():
    return [_poly((-0.6, -0.1), (-0.55, 0.25), (0.3, 0.3), (0.45, 0.1)),         # back
            _poly((-0.6, -0.1), (0.4, -0.1)),                                     # belly
            _circle(0.55, 0.3, 0.17, n=18), _poly((0.48, 0.44), (0.5, 0.56), (0.58, 0.45)),
            _poly((0.62, 0.44), (0.67, 0.55), (0.7, 0.43)),
            _poly((-0.5, -0.1), (-0.52, -0.55)), _poly((-0.3, -0.1), (-0.3, -0.55)),
            _poly((0.15, -0.1), (0.15, -0.55)), _poly((0.35, -0.1), (0.37, -0.55)),
            _circle(-0.8, 0.1, 0.25, n=14, a0=-1.2, a1=1.6)]                    # tail


def m_cat():
    return [_circle(0, -0.15, 0.32, n=24), _circle(0, 0.32, 0.2, n=18),
            _poly((-0.15, 0.45), (-0.2, 0.65), (-0.05, 0.5)), _poly((0.15, 0.45), (0.2, 0.65), (0.05, 0.5)),
            _circle(0.45, -0.25, 0.2, n=12, a0=-1.5, a1=1.5), _poly((-0.06, 0.3), (0.06, 0.3))]


def m_dog():
    return [_poly((-0.5, -0.1), (-0.5, 0.2), (0.3, 0.2), (0.3, -0.1)), _circle(0.45, 0.32, 0.16, n=16),
            _poly((0.35, 0.42), (0.3, 0.6), (0.45, 0.48)),
            _poly((-0.42, -0.1), (-0.42, -0.5)), _poly((-0.2, -0.1), (-0.2, -0.5)),
            _poly((0.05, -0.1), (0.05, -0.5)), _poly((0.25, -0.1), (0.25, -0.5)), _poly((-0.5, 0.15), (-0.7, 0.4))]


def m_bear():
    return [_circle(0, -0.1, 0.45, n=28), _circle(0.2, 0.4, 0.22, n=18), _circle(0.08, 0.58, 0.06, n=8),
            _circle(0.35, 0.58, 0.06, n=8), _poly((-0.25, -0.5), (-0.25, -0.75)), _poly((0.15, -0.5), (0.15, -0.75)),
            _circle(0.26, 0.36, 0.04, n=8)]


def m_fox():
    return [_poly((-0.5, -0.1), (-0.45, 0.2), (0.25, 0.2), (0.35, -0.1)), _poly((0.35, 0.05), (0.7, 0.25), (0.6, 0.45), (0.3, 0.3)),
            _poly((0.35, 0.3), (0.32, 0.5), (0.45, 0.38)), _poly((0.5, 0.36), (0.58, 0.55), (0.6, 0.42)),
            _poly((-0.4, -0.1), (-0.4, -0.45)), _poly((0.2, -0.1), (0.2, -0.45)),
            _circle(-0.75, 0.1, 0.28, n=12, a0=-1.0, a1=1.2)]


def m_snake():
    return [_wave(-0.9, 0.7, 0.05, 0.15, 2, n=40), _wave(-0.9, 0.7, -0.05, 0.15, 2, n=40),
            _circle(0.8, 0.0, 0.1, n=12), _poly((0.9, 0.0), (1.05, 0.05)), _poly((0.9, 0.0), (1.05, -0.05))]


MOTIFS = {
    "sun": m_sun, "moon": m_moon, "star": m_star, "cloud": m_cloud, "rain": m_rain,
    "mountain": m_mountain, "river": m_river, "lake": m_lake, "sea": m_sea,
    "tree": m_tree, "flower": m_flower, "corn": m_corn, "potato": m_potato, "field": m_field,
    "house": m_house, "bridge": m_bridge, "path": m_path, "temple": m_temple, "boat": m_boat,
    "fire": m_fire, "music": m_music,
    "person": m_person, "llama": m_llama, "condor": m_condor, "bird": m_bird, "fish": m_fish,
    "puma": m_puma, "cat": m_cat, "dog": m_dog, "bear": m_bear, "fox": m_fox, "snake": m_snake,
}

SKY = ("sun", "moon", "star", "cloud", "rain", "condor", "bird")
BACK = ("mountain",)
WATER = ("river", "lake", "sea")
WIDE_GROUND = ("field", "path", "bridge")


def all_keys():
    return list(MOTIFS.keys())


def motif(key, cx, cy, size):
    """One motif scaled so that its height is about `size` mm, centred at cx, cy."""
    fn = MOTIFS.get(key)
    if fn is None:
        return []
    return _shift(_scale(fn(), size), cx, cy)


def stepped_border(W, H, inset):
    """Andean stepped (chakana-like) frame: outer stepped rectangle + inner line."""
    s = inset * 0.45          # step size
    x0, y0, x1, y1 = inset, inset, W - inset, H - inset
    outer = [(x0 + s, y0), (x1 - s, y0), (x1 - s, y0 + s), (x1, y0 + s), (x1, y1 - s), (x1 - s, y1 - s),
             (x1 - s, y1), (x0 + s, y1), (x0 + s, y1 - s), (x0, y1 - s), (x0, y0 + s), (x0 + s, y0 + s), (x0 + s, y0)]
    g = s * 0.6
    inner = [(x0 + g + s, y0 + g), (x1 - g - s, y0 + g), (x1 - g - s, y0 + g + s * 0.6), (x1 - g, y0 + g + s * 0.6),
             (x1 - g, y1 - g - s * 0.6), (x1 - g - s, y1 - g - s * 0.6), (x1 - g - s, y1 - g), (x0 + g + s, y1 - g),
             (x0 + g + s, y1 - g - s * 0.6), (x0 + g, y1 - g - s * 0.6), (x0 + g, y0 + g + s * 0.6),
             (x0 + g + s, y0 + g + s * 0.6), (x0 + g + s, y0 + g)]
    # small zigzag ornaments centred on each side
    z = s * 0.35
    top = _zigzag(W / 2 - 4 * s, W / 2 + 4 * s, y1 - g / 2, z * 0.5, 16)
    bottom = _zigzag(W / 2 - 4 * s, W / 2 + 4 * s, y0 + g / 2, z * 0.5, 16)
    return [outer, inner, top, bottom]


def compose(elements, W, H, seed=None):
    """Lay the story's elements out as a scene on a W x H mm sheet."""
    rnd = random.Random(seed)
    els = [e for e in elements if e in MOTIFS]
    if not els:
        els = ["mountain", "sun", "llama"]
    pls = []
    m = min(W, H) * 0.08                                   # margin
    sky = [e for e in els if e in SKY]
    back = [e for e in els if e in BACK]
    water = [e for e in els if e in WATER]
    wide = [e for e in els if e in WIDE_GROUND]
    ground = [e for e in els if e not in SKY + BACK + WATER + WIDE_GROUND]

    # background mountains, wide
    if back:
        pls += motif("mountain", W * 0.55, H * 0.7, H * 0.36)
        pls += motif("mountain", W * 0.15, H * 0.66, H * 0.24)
    # sky items along the top
    n = len(sky)
    for i, e in enumerate(sky):
        x = W * (0.2 + 0.6 * (i + 0.5) / n) if n > 1 else W * (0.82 if back else 0.5)
        y = H * (0.86 if e in ("sun", "moon", "star", "cloud", "rain") else 0.78)
        size = H * (0.16 if e in ("condor",) else 0.13)
        x += rnd.uniform(-W * 0.03, W * 0.03)
        pls += motif(e, x, y, size)
    # water bands
    for e in water:
        if e == "river":
            pls += motif("river", W * 0.5, H * 0.2, W * 0.85)
        elif e == "lake":
            pls += motif("lake", W * 0.5, H * 0.3, W * 0.5)
        else:
            pls += motif("sea", W * 0.5, H * 0.18, W * 0.9)
    for e in wide:
        pls += motif(e, W * 0.5, H * 0.14 if e != "bridge" else H * 0.3, W * (0.8 if e != "bridge" else 0.5))
    # ground subjects in a row
    n = len(ground)
    if n:
        base_y = H * (0.46 if water else 0.34)
        size = min(H * 0.28, (W - 2 * m) / n * 0.75)
        for i, e in enumerate(ground):
            x = m + (W - 2 * m) * (i + 0.5) / n
            y = base_y + rnd.uniform(-H * 0.02, H * 0.02)
            if e in ("tree", "house", "temple", "corn", "flower"):
                y -= size * 0.05
            pls += motif(e, x, y, size)
    # ground line
    pls.append([(m, H * 0.1), (W - m, H * 0.1)])
    return _clamp(pls, W, H, m * 0.5)


def _clamp(pls, W, H, m):
    return [[(min(max(x, m), W - m), min(max(y, m), H - m)) for x, y in pl] for pl in pls]


if __name__ == "__main__":
    for k in all_keys():
        print(k, len(compose([k], 210, 148)))
