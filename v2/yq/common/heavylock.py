"""One-at-a-time lock for memory-heavy work on the 16 GB MacBook.

Loading two big models at once (image generator + VLM + Whisper) swaps the Mac
to a crawl. Anything that loads more than ~1 GB of weights while developing or
testing runs inside this lock:

    from yq.common.heavylock import heavy
    with heavy("flux test"):
        ...

or from a shell, wrapping a whole command:

    python -m yq.common.heavylock pytest -m heavy tests/art
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

LOCK_PATH = Path(os.environ.get("YQ_HEAVY_LOCK", Path.home() / ".cache" / "yq" / "heavy.lock"))


@contextlib.contextmanager
def heavy(label: str = "", poll_s: float = 2.0):
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "a+") as fh:
        waited = 0.0
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if waited == 0.0:
                    print("[heavylock] waiting for another heavy job: %s" % label, file=sys.stderr)
                time.sleep(poll_s)
                waited += poll_s
        try:
            fh.seek(0)
            fh.truncate()
            fh.write("%d %s\n" % (os.getpid(), label))
            fh.flush()
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m yq.common.heavylock <command> [args...]", file=sys.stderr)
        return 2
    with heavy(" ".join(argv)):
        return subprocess.call(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
