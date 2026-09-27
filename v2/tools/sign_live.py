"""Live test of the SignEngine with a camera (Mac webcam or the Arducam on the Jetson).

    .venvs/sign/bin/python tools/sign_live.py --lang prl            # letters
    .venvs/sign/bin/python tools/sign_live.py --lang prl --words    # word signs (needs a words model)
    YQ_MOCK_SIGN_CAMERA=1 .venvs/sign/bin/python tools/sign_live.py # no camera: synthetic signer

Keys: 1-5 = choose candidate, ENTER = accept the best, BACKSPACE = delete, C = clear,
M = letters/words, Q = quit. The terminal prints every token and the fps.
On a Mac the terminal app needs camera permission (System Settings > Privacy > Camera).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yq.sign import SignEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="prl")
    ap.add_argument("--words", action="store_true")
    args = ap.parse_args()
    import cv2
    eng = SignEngine(args.lang, on_token=lambda t: print("TOKEN", t.kind, t.value, round(t.confidence, 2),
                                                        t.alternatives[:2]))
    eng.start()
    eng.set_mode("words" if args.words else "letters")
    last = 0.0
    try:
        while True:
            jpg = eng.latest_jpeg()
            if jpg:
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                st = eng.state()
                cands = "  ".join("%d:%s" % (i + 1, c["text"]) for i, c in enumerate(st["candidates"][:5]))
                cv2.putText(img, eng.text()[-40:], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                cv2.putText(img, cands[:80], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
                cv2.imshow("sign live", img)
                if time.time() - last > 2:
                    last = time.time()
                    print("fps %.1f hands %s buffer %r text %r %s" % (st["fps"], st["hands_visible"], st["buffer"],
                                                                     st["text"], st["error"]))
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                break
            if ord("1") <= k <= ord("5"):
                eng.accept(k - ord("1"))
            elif k == 13:
                eng.accept(0)
            elif k in (8, 127):
                eng.backspace()
            elif k == ord("c"):
                eng.clear()
            elif k == ord("m"):
                eng.set_mode("words" if eng.mode == "letters" else "letters")
    finally:
        eng.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
