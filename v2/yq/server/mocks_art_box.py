"""UI-side mocks for ART (make_drawing) and BOX (preflight, run_scan).

The box mock writes a real scan folder (CONTRACTS.md §4 layout) with synthetic
photos, a GLB model, PTM relighting data, UV and thermal images, so every
results viewer of the kiosk works without the box. Results say engine "mock".
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from yq.common import config
from yq.common.contracts import DrawingResult, Progress, ScanResult, StoryInput
from yq.server import mock_assets as A
from yq.server import settings


def _sleep(s: float, stop: Optional[threading.Event] = None) -> bool:
    s = max(0.0, s * float(settings.MOCK_SPEED))
    return stop.wait(s) if stop is not None else (time.sleep(s) or False)


def _emit(cb, stage, fraction, msg, **detail):
    if cb:
        cb(Progress(stage=stage, fraction=round(float(fraction), 4), message_es=msg, detail=detail))


class MockArt:
    source = "mock:ui"
    fail = False                       # tests: make the next drawing fail

    def make_drawing(self, story: StoryInput, out_dir: Path, on_progress=None, published: bool = True) -> DrawingResult:
        from yq.publish.gallery import story_url
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _emit(on_progress, "planning", 0.05, "Pensando qué dibujar de tu historia…")
        _sleep(1.2)
        if MockArt.fail:
            MockArt.fail = False
            raise RuntimeError("mock drawing failure")
        for i in range(4):
            _emit(on_progress, "generating", 0.15 + i * 0.13, "Imaginando el dibujo…", attempt=1, step=i + 1, steps=4)
            _sleep(0.8)
        lines = A.drawing_polylines(seed=len(story.text))
        A.write(out_dir / "image.png", A.drawing_png(lines))
        verified = {"condor": True, "mountains": True, "llama": True, "lake": True}
        _emit(on_progress, "verifying", 0.72, "Revisando que el dibujo tenga todo…",
              image=str(out_dir / "image.png"), verified=verified)
        _sleep(1.0)
        _emit(on_progress, "vectorizing", 0.85, "Convirtiendo el dibujo en trazos de lápiz…")
        _sleep(0.8)
        qr = story_url(story.story_id) if published else config.PUBLIC_BASE_URL   # declined: general gallery
        title = " ".join((story.text_es or story.text).split()[:5]).rstrip(".,;:") + "…"
        A.write(out_dir / "front.svg", A.svg_paths(lines))
        A.write(out_dir / "back.svg", A.back_svg(title, story.text_es or story.text, qr))
        _emit(on_progress, "layout", 0.95, "Preparando el reverso con tu historia y el QR…")
        _sleep(0.5)
        pen = sum(sum(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 for (x1, y1), (x2, y2) in zip(pl, pl[1:])) for pl in lines)
        return DrawingResult(story_id=story.story_id, image=str(out_dir / "image.png"),
                             front_svg=str(out_dir / "front.svg"), back_svg=str(out_dir / "back.svg"),
                             verified=verified, attempts=1, strokes=len(lines), pen_mm=round(pen, 1),
                             est_minutes=round(pen / 1500.0 + len(lines) * 0.02, 1), qr_url=qr)


IDENTIFICATION = {
    "object_type": "stirrup-spout vessel", "object_type_es": "Vasija de asa estribo",
    "material": "fired clay with slip paint", "material_es": "Cerámica (arcilla cocida) con engobe pintado",
    "culture": "Moche", "period": "100–800 d. C.", "region": "Costa norte del Perú", "confidence": 0.62,
    "alternatives": [
        {"culture": "Chimú", "period": "900–1470 d. C.", "confidence": 0.21,
         "why": "También hacían asa estribo, pero casi siempre negra y bruñida."},
        {"culture": "Cupisnique", "period": "1500–500 a. C.", "confidence": 0.09,
         "why": "Asa estribo más gruesa y decoración incisa, sin pintura crema."}],
    "evidence": ["Tiene asa estribo con pico: forma típica de la costa norte.",
                 "Pintura crema y rojo oscuro (dos colores), como en la cerámica Moche.",
                 "Banda con escalones pintados, un motivo andino muy usado."],
    "similar": [
        {"title": "Botella de asa estribo con banda escalonada", "culture": "Moche", "date": "200–800 d. C.",
         "museum": "Catálogo de ejemplo (simulado)", "url": "", "image": "", "score": 0.81},
        {"title": "Botella de asa estribo negra", "culture": "Chimú", "date": "1100–1470 d. C.",
         "museum": "Catálogo de ejemplo (simulado)", "url": "", "image": "", "score": 0.64}],
    "description_es": "Parece una botella ceremonial de asa estribo, pintada en crema y rojo. "
                      "Esta forma se usó durante siglos en la costa norte del Perú.",
    "engine": "mock",
}

MEASUREMENTS = [
    {"name": "mass", "value": 312.4, "unit": "g", "uncertainty": 0.4, "method": "balanza (celda de carga)"},
    {"name": "height", "value": 136.0, "unit": "mm", "uncertainty": 1.2, "method": "modelo 3D"},
    {"name": "width", "value": 100.3, "unit": "mm", "uncertainty": 1.2, "method": "modelo 3D"},
    {"name": "depth", "value": 97.8, "unit": "mm", "uncertainty": 1.2, "method": "modelo 3D"},
    {"name": "volume_envelope", "value": 612.0, "unit": "cm3", "uncertainty": 45.0, "method": "modelo 3D",
     "note": "Incluye el hueco interior."},
    {"name": "density_apparent", "value": 0.51, "unit": "g/cm3", "uncertainty": 0.04, "method": "masa / volumen"},
]


NORTH_COAST_WORDS = ("trujillo", "chiclayo", "lambayeque", "piura", "moche", "sipán", "sipan", "huaca", "chan chan")


def identify_with_context(ctx: dict) -> dict:
    """Mock of CONTRACTS §9: identify from the image first, then use the place only as a bounded clue."""
    ident = dict(IDENTIFICATION)
    image_only = {"culture": "Moche", "period": "100–800 d. C.", "material": ident["material_es"], "confidence": 0.48}
    text = (ctx.get("found_where") or "").lower()
    region = ctx.get("region_hint") or ""
    if not text and region in ("", "no_se"):
        ident.update(confidence=0.48, context_effect_es="", image_only=None)
        return ident
    ident["image_only"] = image_only
    if "chan chan" in text or "chimú" in text or "chimu" in text:
        ident.update(culture="Chimú", period="900–1470 d. C.", confidence=0.57, alternatives=[
            {"culture": "Moche", "period": "100–800 d. C.", "confidence": 0.33,
             "why": "La imagen se parece un poco más a Moche."}] + ident["alternatives"][1:],
            context_effect_es="El lugar (Chan Chan) inclinó la balanza hacia Chimú: la imagen dudaba entre Moche y Chimú.")
    elif region == "costa_norte" or any(w in text for w in NORTH_COAST_WORDS):
        ident.update(confidence=0.62,
                     context_effect_es="El lugar coincide con la costa norte y ayudó a decidir entre Moche y Chimú.")
    elif region in ("selva", "altiplano", "otro_pais"):
        ident.update(confidence=0.48,
                     context_effect_es="El lugar no coincide con lo que se ve; se priorizó la imagen.")
    else:
        ident.update(confidence=0.48, context_effect_es="El lugar no cambió la respuesta.")
    return ident


class MockBox:
    source = "mock:ui"
    problems: list = []                # tests: preflight problems to report
    fail_at: str = ""                  # tests: stage name that raises

    def preflight(self) -> dict:
        _sleep(0.8)
        if MockBox.problems:
            return {"ok": False, "problems_es": list(MockBox.problems), "weight_g": None}
        return {"ok": True, "problems_es": [], "weight_g": 312.4}

    def run_scan(self, req, on_progress=None, cancel_event: Optional[threading.Event] = None) -> ScanResult:
        cancel = cancel_event or threading.Event()
        folder = Path(config.SCANS_DIR) / req.scan_id
        started = time.time()
        res = ScanResult(scan_id=req.scan_id, folder=str(folder), profile=req.profile, started=started)
        A.write(folder / "meta.json", json.dumps({"scan_id": req.scan_id, "profile": req.profile,
                                                  "analyses": list(req.analyses), "started": started,
                                                  "context": dict(getattr(req, "context", None) or {}),
                                                  "door_closed": True, "warnings": [], "source": "mock"}))

        def step(stage, frac, msg, secs, **detail):
            if MockBox.fail_at == stage:
                MockBox.fail_at = ""
                raise RuntimeError("mock box failure at %s" % stage)
            _emit(on_progress, stage, frac, msg, **detail)
            if _sleep(secs, cancel):
                raise _Cancelled()

        try:
            step("weighing", 0.03, "Pesando el objeto…", 1.0, weight_g=312.4)
            A.write(folder / "weight.json", json.dumps({"grams": 312.4, "sigma_g": 0.4, "stable": True}))
            angles = list(range(0, 360, 15))
            for i, a in enumerate(angles):
                for cam in ("A", "B"):
                    rel = "photogrammetry/cam%s_%03d.jpg" % (cam, a)
                    A.write(folder / rel, A.object_photo(a + (0 if cam == "A" else 7), "white", 320, 240))
                    step("photogrammetry", 0.05 + 0.4 * (i * 2 + (cam == "B")) / (2 * len(angles)),
                         "Girando el plato y tomando fotos: %d°" % a, 0.17,
                         platter_deg=float(a), camera=cam, light="cob", photo=rel, total=len(angles))
            for k in range(1, 9):
                rel = "rti/led%d.jpg" % k
                A.write(folder / rel, A.object_photo(0, "led%d" % k, 320, 240))
                step("rti", 0.45 + 0.15 * k / 8, "Luz %d de 8: buscando relieves" % k, 0.3,
                     light="led%d" % k, led_index=k, leds=8, photo=rel)
            A.write(folder / "uv/uv.jpg", A.object_photo(0, "uv"))
            step("uv", 0.63, "Luz ultravioleta: buscando restauraciones", 0.8, light="uv", photo="uv/uv.jpg")
            A.write(folder / "uv/visible.jpg", A.object_photo(0, "white"))
            step("uv", 0.66, "Foto con luz blanca para comparar", 0.4, light="cob", photo="uv/visible.jpg")
            for j in range(12):
                t = (j + 1) / 6.0
                rel = "thermal/preview_%02d.png" % j
                A.write(folder / rel, A.thermal_png(t))
                heating = j < 6
                tmax = round(float(A.thermal_field(t).max()), 1)
                step("thermal", 0.66 + 0.19 * (j + 1) / 12,
                     "Calentando suavemente…" if heating else "Mirando cómo se enfría…", 0.45,
                     light="halogen" if heating else "off", phase="heating" if heating else "cooling",
                     thermal_preview=rel, temp_max_c=tmax)
            for k, msg in enumerate(("Armando el modelo 3D…", "Calculando la luz rasante (RTI)…",
                                     "Comparando con piezas de museo…")):
                step("analyzing", 0.87 + 0.04 * k, msg, 1.2, where="local")
        except _Cancelled:
            res.ok, res.finished, res.warnings = False, time.time(), ["cancelado"]
            return res

        an = folder / "analysis"
        A.write(an / "model.glb", A.vessel_glb())
        A.ptm_files(an / "rti")
        A.write(an / "uv_overlay.jpg", A.uv_overlay())
        A.write(an / "thermal_max.png", A.thermal_result("max"))
        A.write(an / "thermal_anomaly.png", A.thermal_result("anomaly"))
        res.artifacts = {"model_glb": "analysis/model.glb", "rti_ptm": "analysis/rti/ptm.json",
                         "uv_image": "uv/uv.jpg", "visible_image": "uv/visible.jpg",
                         "uv_overlay": "analysis/uv_overlay.jpg", "thermal_max": "analysis/thermal_max.png",
                         "thermal_anomaly": "analysis/thermal_anomaly.png",
                         "photo_front": "photogrammetry/camA_000.jpg"}
        res.findings = [
            {"analysis": "uv", "title_es": "Una zona brilla distinto con luz UV",
             "detail_es": "A la derecha del cuerpo hay una mancha que brilla verde-amarilla. Puede ser pegamento "
                          "o una restauración moderna.", "severity": "warning", "image": "analysis/uv_overlay.jpg"},
            {"analysis": "thermal", "title_es": "Una zona se enfría más lento",
             "detail_es": "Puede haber una grieta por dentro o un relleno de otro material.",
             "severity": "warning", "image": "analysis/thermal_anomaly.png"},
            {"analysis": "rti", "title_es": "Relieve escalonado y espiral",
             "detail_es": "Con luz rasante se ven líneas grabadas que casi no se notan en la foto normal.",
             "severity": "info", "image": ""},
        ]
        res.measurements = [dict(m) for m in MEASUREMENTS]
        res.identification = identify_with_context(getattr(req, "context", None) or {})
        res.warnings = ["Resultado SIMULADO (modo de prueba, sin caja real)."]
        _emit(on_progress, "done", 1.0, "¡Listo!")
        res.finished = time.time()
        from yq.common.contracts import to_dict
        A.write(folder / "result.json", json.dumps(to_dict(res), ensure_ascii=False, indent=1))
        return res


class _Cancelled(Exception):
    pass
