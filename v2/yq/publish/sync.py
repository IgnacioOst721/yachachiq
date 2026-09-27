"""Store-and-forward publishing of stories to the web gallery (GitHub Pages), ported from v1.

Every story lives in config.STORIES_DIR/<story_id>/ (story.json, story.txt, scene_1.png,
storyteller_photo.jpg, ...). sync() copies the finished public ones into
<archive>/docs/stories/<web_id>/, rebuilds docs/stories.json, commits and pushes to
`main` (GitHub Pages serves main:/docs). The archive is a SEPARATE clone
(~/yachachiq-archivo) so publishing never touches the code checkout.

No internet -> nothing happens: stories wait on disk and go up on the next sync()
(after every story, every few minutes, after boot). The disk is the source of truth:
the archive clone is reset to origin/main before every publish, so a push that failed
halfway can never wedge publishing. Markers per story: `.ready` (finished + consent),
`.private` (visitor said no), `.published` (on the web). Never raises.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from yq.common import config
from yq.publish import gallery, settings

log = logging.getLogger("yq.publish")
_lock = threading.Lock()
_again = {"flag": False}
last_result: dict = {"status": "never", "new": 0}


def mode() -> str:
    if not settings.PUBLISH_ENABLED:
        return "off"
    if config.mock("publish") or settings.ui_mocked():
        return "mock"
    return "git"


def _git(args, cwd, timeout, check=False):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")        # never hang on a password prompt
    return subprocess.run(["git"] + list(args), cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout, check=check, env=env)


def have_internet(repo: Path, timeout: float = 8) -> bool:
    try:
        _git(["ls-remote", "--exit-code", "-h", settings.PUBLISH_REMOTE, settings.PUBLISH_BRANCH],
             repo, timeout, check=True)
        return True
    except Exception:
        return False


def _story_dirs(stories_dir: Path) -> list:
    if not stories_dir.is_dir():
        return []
    return sorted(d for d in os.listdir(stories_dir)
                  if (stories_dir / d).is_dir() and (stories_dir / d / "story.txt").exists())


def pending(stories_dir=None) -> list:
    """Finished public stories not yet on the web."""
    sd = Path(stories_dir or config.STORIES_DIR)
    return [d for d in _story_dirs(sd)
            if (sd / d / ".ready").exists() and not (sd / d / ".private").exists()
            and not (sd / d / ".published").exists()]


def _meta(folder: Path) -> dict:
    try:
        return json.loads((folder / "story.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _publish_once(repo: Path, stories_dir: Path, names: list) -> int:
    branch, remote = settings.PUBLISH_BRANCH, settings.PUBLISH_REMOTE
    web = repo / "docs" / "stories"
    _git(["fetch", "-q", remote, branch], repo, 90, check=True)
    _git(["reset", "-q", "--hard", "%s/%s" % (remote, branch)], repo, 30, check=True)
    _git(["clean", "-fdq", "docs"], repo, 30)
    web.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = stories_dir / name
        dst = web / gallery.web_id(name)
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in gallery.WEB_FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        (dst / "info.json").write_text(json.dumps(gallery.info_json(_meta(src)), ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        redirect = repo / "docs" / name                  # the printed QR: PUBLIC_BASE_URL + story_id
        redirect.mkdir(parents=True, exist_ok=True)
        (redirect / "index.html").write_text(gallery.redirect_html(name), encoding="utf-8")
    (repo / "docs" / "stories.json").write_text(
        json.dumps(gallery.index(web), ensure_ascii=False, indent=1), encoding="utf-8")
    _git(["add", "-f", "docs"], repo, 30, check=True)      # -f: a stray .gitignore once hid every image
    ident = []
    if not _git(["config", "user.email"], repo, 10).stdout.strip():
        ident = ["-c", "user.name=Yachachiq", "-c", "user.email=yachachiq@robot.local"]
    r = _git(ident + ["commit", "-q", "-m", "Publicar %d historia(s)" % len(names)], repo, 30)
    if "nothing to commit" in (r.stdout + r.stderr):
        return 0
    if r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, "git commit", r.stdout, r.stderr)
    _git(["push", "-q", remote, "HEAD:%s" % branch], repo, 180, check=True)
    return len(names)


def sync(stories_dir=None, repo=None) -> dict:
    """Returns {"status": off|mock|nothing|offline|published|error, "new": n, ...}."""
    global last_result
    res = _sync(Path(stories_dir or config.STORIES_DIR), Path(repo or settings.PUBLISH_REPO_DIR))
    res["t"] = time.time()
    last_result = res
    return res


def _sync(stories_dir: Path, repo: Path) -> dict:
    m = mode()
    if m == "off":
        return {"status": "off", "new": 0}
    names = pending(stories_dir)
    if m == "mock":
        return {"status": "mock", "new": len(names)}
    if not names:
        return {"status": "nothing", "new": 0}
    if not (repo / ".git").exists():
        return {"status": "error", "new": 0, "error": "no hay archivo web (%s)" % repo}
    if not have_internet(repo):
        return {"status": "offline", "new": len(names)}
    try:
        try:
            new = _publish_once(repo, stories_dir, names)
        except subprocess.CalledProcessError as e:
            log.info("publish: retrying after git error: %s", (e.stderr or "")[:200])
            new = _publish_once(repo, stories_dir, names)
        for name in names:
            (stories_dir / name / ".published").write_text(
                "%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), gallery.story_url(name)), encoding="utf-8")
        log.info("published %d new stories", new)
        return {"status": "published", "new": new, "ids": [gallery.web_id(n) for n in names]}
    except Exception as e:
        err = getattr(e, "stderr", "") or str(e)
        log.warning("publish failed: %s", str(err)[:300])
        return {"status": "error", "new": len(names), "error": str(err)[:200]}


def sync_later(on_done: Optional[Callable[[dict], None]] = None, stories_dir=None, repo=None) -> dict:
    """Publish in a background thread; returns at once with what is queued."""
    n = len(pending(stories_dir)) if mode() != "off" else 0
    if mode() in ("off", "mock") or not n:
        res = sync(stories_dir, repo)
        if on_done:
            on_done(res)
        return res

    def work():
        while True:
            _again["flag"] = False
            res = sync(stories_dir, repo)
            if on_done:
                try:
                    on_done(res)
                except Exception:
                    log.exception("publish on_done failed")
            if not _again["flag"]:
                break

    if _lock.acquire(blocking=False):
        def run():
            try:
                work()
            finally:
                _lock.release()
        threading.Thread(target=run, name="publish", daemon=True).start()
    else:
        _again["flag"] = True                  # a sync is running: it will go round once more
    return {"status": "queued", "new": n}


def status(stories_dir=None) -> dict:
    sd = Path(stories_dir or config.STORIES_DIR)
    dirs = _story_dirs(sd)
    return {"mode": mode(), "pending": len(pending(sd)),
            "published": sum(1 for d in dirs if (sd / d / ".published").exists()),
            "private": sum(1 for d in dirs if (sd / d / ".private").exists()),
            "repo": str(settings.PUBLISH_REPO_DIR), "last": last_result}
