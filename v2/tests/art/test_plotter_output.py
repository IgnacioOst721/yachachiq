"""Pen order, text, QR, SVG and G-code: everything the printer receives."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from yq.art import gcode, geom, hatch, layout, order, qr, settings, svg, text


def _random_strokes(n=400, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = rng.uniform(20, 180, 2)
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        out.append(np.array([p, p + d * rng.uniform(1, 10)]))
    return out


def test_order_reduces_travel_and_keeps_ink():
    S = _random_strokes()
    greedy = order._nn(S, (0, 0))
    opt = order.optimize(S, (0, 0), budget_s=2.0)
    t_naive, t_greedy, t_opt = geom.travel(S), geom.travel(greedy), geom.travel(opt)
    assert t_opt < t_greedy <= t_naive
    assert t_opt < 0.2 * t_naive
    assert abs(geom.total_length(opt) - geom.total_length(S)) < 1e-6
    # every original stroke is drawn exactly once (maybe reversed)
    key = lambda s: tuple(sorted([tuple(np.round(s[0], 6)), tuple(np.round(s[-1], 6))]))
    assert sorted(map(key, S)) == sorted(map(key, opt))


def test_two_opt_fixes_a_crossing_tour():
    # four strokes in a row, deliberately visited out of order
    S = [np.array([[x, 0.0], [x + 1, 0.0]]) for x in (0, 30, 10, 20)]
    opt = order.optimize(S, (0, 0), budget_s=1.0)
    assert geom.travel(opt) <= 30.0 + 1e-6


def test_join_touching_strokes():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [2.0, 1.0]])
    assert len(order.join_touching([a, b])) == 1


def test_continuous_route_draws_everything():
    S = order.optimize(_random_strokes(120), (0, 0), 1.0)
    route, visible = order.continuous(S, (0, 0))
    assert len(route) == 1
    R = route[0]
    for s in S:                                     # every stroke point lies on the route
        assert geom.point_segment_dist(s, [R]).max() < 1e-6
    assert visible < geom.travel(S) + 1e-6


def test_hershey_wrap_accents_and_margins():
    story = ("¿Sabías que el cóndor andino vuela sobre los Apus? ¡Sí! Ñawpa pachapi huk kuntursi karqan, "
             "wik'uñakunata qhawaspa. Pingüinos y ñandúes no viven aquí.") * 3
    x0, y0, x1, y1 = settings.printable_box()
    st, h, lines = text.paragraph(story, x0, y0, x1 - x0, 4.0)
    assert len(lines) > 3
    for ln in lines:
        assert text.width(ln, 4.0) <= (x1 - x0) + 1e-6
    b = geom.bounds(st)
    assert b[0] >= x0 - 0.5 and b[2] <= x1 + 0.5 and b[1] >= y0 - 1.5
    joined = " ".join(lines)
    for ch in "¿¡ñÑóíúü'":
        assert ch in joined
    assert text.supported_fraction(story) == 1.0
    assert text.supported_fraction("我喜欢秃鹰") < 0.5


def test_accented_glyphs_have_marks():
    base = len(text.render_line("n", 0, 0, 5))
    assert len(text.render_line("ñ", 0, 0, 5)) == base + 1
    assert len(text.render_line("é", 0, 0, 5)) == len(text.render_line("e", 0, 0, 5)) + 1
    assert len(text.render_line("ü", 0, 0, 5)) == len(text.render_line("u", 0, 0, 5)) + 2
    # dotless i under the accent: the dot is replaced, not stacked
    assert len(text.render_line("í", 0, 0, 5)) == len(text.render_line("i", 0, 0, 5))
    # ¿ hangs below the baseline like in print
    assert geom.bounds(text.render_line("¿", 0, 0, 5))[3] > 0.5


def test_fit_paragraph_truncates_long_text():
    long = "palabra " * 2000
    st, size, h, fitted = text.fit_paragraph(long, 10, 10, 100, 50, 4.0, 2.5)
    assert not fitted and h <= 50 and size == 2.5


@pytest.mark.parametrize("pen", [0.3, 0.5, 0.7])
def test_qr_decodes_from_pen_strokes(pen):
    url = qr.story_url("story-20260927-101500-ab12")
    assert url.startswith("http")
    st, info = qr.strokes(url, 120, 200, 32.0, pen)
    assert qr.decode_strokes(st, 120, 200, 32.0, pen) == url
    b = geom.bounds(st)
    assert b[2] - b[0] <= 32.0 + 1e-6 and b[2] - b[0] >= 30.0


def test_qr_opencv_fallback_matrix(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "segno":
            raise ImportError("no segno")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    url = qr.story_url("story-x")
    st, info = qr.strokes(url, 10, 10, 36.0, 0.5)
    assert qr.decode_strokes(st, 10, 10, 36.0, 0.5) == url


def test_hatch_rect_serpentine_covers_rect():
    s = hatch.hatch_rect(0, 0, 10, 4, 0.5, pen=0.5)
    assert s[:, 1].min() >= 0.25 - 1e-9 and s[:, 1].max() <= 3.75 + 1e-9
    assert len(s) >= 2 * 7


def test_back_page_layout_within_margins():
    t = "Huk p'unchawsi huk kuntur urqu patapi tiyarqan. " * 6
    es = "Un día un cóndor se sentó en la cima de la montaña. " * 6
    st, info = layout.back_page(t, "quy_Latn", es, "Kunturmanta", qr.story_url("story-1"))
    x0, y0, x1, y1 = settings.printable_box()
    b = geom.bounds(st)
    assert b[0] >= x0 - 0.6 and b[1] >= y0 - 0.6 and b[2] <= x1 + 0.6 and b[3] <= y1 + 0.6
    assert info["parts"] == 2 and settings.TEXT_MIN_MM <= info["text_mm"] <= settings.TEXT_MAX_MM


def test_back_page_non_latin_story_prints_spanish():
    st, info = layout.back_page("从前有一只秃鹰。", "cmn_Hans", "Había una vez un cóndor.", "El cóndor", "")
    assert info["parts"] == 2 and len(st) > 10


def test_gcode_bounds_and_modes(monkeypatch):
    S = [np.array([[10.0, 10.0], [200.0, 287.0]]), np.array([[105.0, 148.0]])]
    for mode in ("z", "servo", "none"):
        monkeypatch.setattr(settings, "PEN_MODE", mode)
        lines = gcode.to_gcode(S, "t")
        xs = [float(w[1:]) for ln in lines if ln.startswith(("G0", "G1")) for w in ln.split() if w[0] == "X"]
        ys = [float(w[1:]) for ln in lines if ln.startswith(("G0", "G1")) for w in ln.split() if w[0] == "Y"]
        assert min(xs) >= 0 and max(xs) <= settings.PAPER_W_MM
        assert min(ys) >= 0 and max(ys) <= settings.PAPER_H_MM
        assert ("M3 S%d" % settings.SERVO_DOWN in lines) == (mode == "servo")
        assert any(ln.startswith("G1 Z") for ln in lines) == (mode == "z")
    # y is flipped: page top (y=10) is machine Y = 287
    monkeypatch.setattr(settings, "PEN_MODE", "z")
    assert "G0 X10.00 Y287.00" in gcode.to_gcode(S)
    with pytest.raises(ValueError):
        gcode.to_gcode([np.array([[0.0, 0.0], [300.0, 10.0]])])


def test_time_estimate_is_sane(monkeypatch):
    monkeypatch.setattr(settings, "DRAW_FEED", 1200.0)
    monkeypatch.setattr(settings, "ACCEL_MM_S2", 1e6)
    monkeypatch.setattr(settings, "PEN_MODE", "none")
    e = gcode.estimate([np.array([[0.0, 297.0], [200.0, 297.0]])], start=(0.0, 297.0))
    assert abs(e["seconds"] - 10.0) < 0.2           # 200 mm at 20 mm/s
    monkeypatch.setattr(settings, "ACCEL_MM_S2", 10.0)
    zig = np.array([[x, 100.0 + (x % 2)] for x in range(0, 200)], dtype=float)
    assert gcode.estimate([zig])["seconds"] > 60    # zig-zags at low acceleration are slow


def test_svg_is_wellformed_mm_one_path_per_stroke():
    S = [np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 10.0]]), np.array([[50.0, 50.0]])]
    txt = svg.to_svg({"black": S, "red": S[:1]})
    root = ET.fromstring(txt.split("?>", 1)[1])
    assert root.attrib["width"] == "210mm" and root.attrib["height"] == "297mm"
    assert root.attrib["viewBox"] == "0 0 210 297"
    ns = "{http://www.w3.org/2000/svg}"
    groups = root.findall(ns + "g")
    assert len(groups) == 2
    assert len(groups[0].findall(ns + "path")) == 2 and len(groups[1].findall(ns + "path")) == 1
