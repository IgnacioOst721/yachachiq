"""Grabador guiado de señas (palabras o letras) con keypoints RTMW.

    python -m yq.sign.record --lang prl --mode words --signer ignacio --reps 5
    python -m yq.sign.record --lang prl --mode words --signer ana --words "cóndor,llama,sol"
    python -m yq.sign.record --lang prl --mode letters --signer ignacio --letters abcdefghijklmnñopqrstuvwxyz

Keys: ESPACIO = grabar, S = saltar, R = borrar la última toma, Q/ESC = salir.
Each take -> MODELS_DIR/sign/recordings/<lang>/<mode>/<label>/<signer>_<session>_NNN.npz
(canonical 69 keypoints + the full 133 RTMW points, times, labels, quality report).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import features as F
from . import recording as R
from . import skeleton as sk
from . import settings, sources

FONT_CANDIDATES = ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_fonts: dict = {}


def put_text(img, text: str, xy, size: int = 32, color=(255, 255, 255)):
    """Unicode text (tildes, ñ) with PIL; OpenCV's putText only draws ASCII."""
    from PIL import Image, ImageDraw, ImageFont
    if size not in _fonts:
        f = None
        for p in FONT_CANDIDATES:
            if Path(p).exists():
                f = ImageFont.truetype(p, size)
                break
        _fonts[size] = f or ImageFont.load_default()
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    d.text((xy[0] + 2, xy[1] + 2), text, font=_fonts[size], fill=(0, 0, 0))
    d.text(xy, text, font=_fonts[size], fill=(color[2], color[1], color[0]))
    img[:] = np.array(pil)[:, :, ::-1]
    return img


class Recorder:
    def __init__(self, args):
        from .pose import PoseEstimator
        self.args = args
        self.cam = sources.open_camera(args.camera)
        self.pose = PoseEstimator(hand_refine="letters" if args.mode == "letters" else "off")
        self.session = args.session or time.strftime("%Y%m%d-%H%M")
        self.last_saved = None

    def frame(self):
        ok, img, t, _n = self.cam.read()
        if not ok:
            return None, None
        pf = self.pose.process(img, t)
        return img, pf

    def show(self, img, pf, lines, color=(255, 255, 255), big=None):
        import cv2
        view = img.copy()
        if pf is not None:
            sources.draw_skeleton(view, pf.canon_xy, pf.canon_conf)
        if settings.SIGN_MIRROR:
            view = cv2.flip(view, 1)
        y = 20
        for i, line in enumerate(lines):
            put_text(view, line, (20, y), 34 if i == 0 else 26, color if i == 0 else (230, 230, 230))
            y += 44 if i == 0 else 34
        if big:
            put_text(view, big, (view.shape[1] // 2 - 40, view.shape[0] // 2 - 80), 140, (0, 220, 255))
        cv2.imshow("Yachachiq - grabar señas", view)
        return cv2.waitKey(1) & 0xFF

    def take(self, label: str, idx: str) -> str:
        """One take of `label`. Returns 'ok', 'skip', 'quit' or 'again'."""
        a = self.args
        motion = a.mode == "letters" and label in R.MOTION_LETTERS_ALL
        how = ("Haz la seña completa y luego BAJA las manos." if a.mode == "words" else
               ("Haz la letra CON su movimiento, dos veces." if motion else "Mantén la letra QUIETA."))
        while True:                                    # wait for SPACE
            img, pf = self.frame()
            if img is None:
                continue
            k = self.show(img, pf, ["%s  «%s»  (%s)" % ("Palabra" if a.mode == "words" else "Letra", label.upper(), idx),
                                    how, "ESPACIO = grabar   S = saltar   R = borrar la última   Q = salir"])
            if k == ord(" "):
                break
            if k in (ord("s"), ord("S")):
                return "skip"
            if k in (ord("q"), ord("Q"), 27):
                return "quit"
            if k in (ord("r"), ord("R")) and self.last_saved and self.last_saved.exists():
                self.last_saved.unlink()
                print("borrada", self.last_saved)
                self.last_saved = None
        t_end = time.monotonic() + 3.0                  # countdown
        while time.monotonic() < t_end:
            img, pf = self.frame()
            if img is not None:
                self.show(img, pf, ["Prepárate...", how], big=str(int(t_end - time.monotonic()) + 1))
        xs, cs, ts, ks, ss = [], [], [], [], []
        dur = a.seconds if a.seconds else (2.0 if a.mode == "letters" else 4.0)
        t0 = time.monotonic()
        quiet = None
        while time.monotonic() - t0 < dur:
            img, pf = self.frame()
            if img is None:
                continue
            xs.append(pf.canon_xy)
            cs.append(pf.canon_conf)
            ts.append(pf.t)
            ks.append(pf.kpts)
            ss.append(pf.scores)
            self.show(img, pf, ["● GRABANDO «%s»" % label.upper(), how], color=(0, 0, 255))
            if a.mode == "words" and time.monotonic() - t0 > 0.8:      # stop early when hands rest
                side_ok = any(F.hand_usable(pf.canon_conf[s]) for s in (sk.C_LHAND, sk.C_RHAND))
                quiet = (quiet or time.monotonic()) if not side_ok else None
                if quiet and time.monotonic() - quiet > 0.5:
                    break
        xy, cf, t = np.stack(xs), np.stack(cs), np.array(ts)
        ok, problems, stats = R.check_quality(xy, cf, t, a.mode, label)
        if not ok:
            for _ in range(60):
                img, pf = self.frame()
                if img is not None:
                    self.show(img, pf, ["Repetimos: " + problems[0]] + problems[1:], color=(0, 140, 255))
            return "again"
        side = "right" if np.mean(cf[:, 48]) >= np.mean(cf[:, 27]) else "left"
        path = R.take_path(a.lang, a.mode, label, a.signer, self.session)
        R.save_take(path, xy, cf, t, label, a.signer, self.session, a.lang, a.mode, pf.image_wh, side,
                    self.pose.model_name, stats, np.stack(ks), np.stack(ss))
        self.last_saved = path
        print("guardado", path, stats)
        return "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grabar señas para entrenar Yachachiq")
    ap.add_argument("--lang", default="prl", choices=["prl", "ase", "ils"])
    ap.add_argument("--mode", default="words", choices=["words", "letters"])
    ap.add_argument("--signer", required=True, help="nombre o apodo de quien seña (sin tildes)")
    ap.add_argument("--session", default=None)
    ap.add_argument("--words", default=None, help="lista separada por comas (por defecto vocab_<lang>.txt)")
    ap.add_argument("--letters", default="abcdefghijklmnñopqrstuvwxyz")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--camera", default=None)
    a = ap.parse_args(argv)
    if a.mode == "letters":
        labels = list(a.letters)
    elif a.words:
        labels = [w.strip() for w in a.words.split(",") if w.strip()]
    else:
        vf = Path(__file__).with_name("vocab_%s.txt" % a.lang)
        labels = [l.strip() for l in vf.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    rec = Recorder(a)
    try:
        for rep in range(a.reps):
            for i, label in enumerate(labels):
                while True:
                    r = rec.take(label, "%d/%d, vez %d de %d" % (i + 1, len(labels), rep + 1, a.reps))
                    if r != "again":
                        break
                if r == "quit":
                    return 0
    finally:
        import cv2
        rec.cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
