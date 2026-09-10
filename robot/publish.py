"""Store-and-forward publishing of stories to the web gallery (GitHub Pages).

Every story lives in config.STORIES_DIR/<timestamp>/ (story.txt, scene_1.png, scene_1_photo.png,
storyteller_photo.jpg). sync() copies the new ones into <archive>/docs/stories/, rebuilds
docs/stories.json, commits and pushes to `main` (GitHub Pages serves main:/docs).

No internet -> nothing happens: the stories wait on the SD card and go up the next time sync()
runs (after every story, every few minutes, and after boot). The SD card is the source of truth:
the archive clone is reset to origin/main before every publish, so a push that failed halfway can
never wedge publishing for good. A story that made it to the web gets a `.published` marker; one
the storyteller kept private has `.private`; only stories with `.ready` (finished) are considered. Never raises.
"""
import json
import logging
import os
import shutil
import subprocess
import threading
import time

import config

log = logging.getLogger("publish")
_lock = threading.Lock()
_again = {"flag": False}


def mode():
    if not config.PUBLISH_ENABLED:
        return "off"
    if config.MOCK:
        return "mock"
    return "git"


def _git(args, cwd, timeout, check=False):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=check)


def have_internet(timeout=6):
    try:
        _git(["ls-remote", "--exit-code", "-h", "origin", "main"], str(config.PUBLISH_REPO_DIR), timeout, check=True)
        return True
    except Exception:
        return False


def _story_dirs(stories_dir):
    if not os.path.isdir(stories_dir):
        return []
    return sorted(d for d in os.listdir(stories_dir)
                  if os.path.isdir(os.path.join(stories_dir, d))
                  and os.path.exists(os.path.join(stories_dir, d, "story.txt")))


def pending(stories_dir=None):
    """Finished stories (`.ready`: photo taken and consent given) that are not on the web yet."""
    stories_dir = str(stories_dir or config.STORIES_DIR)
    return [d for d in _story_dirs(stories_dir)
            if os.path.exists(os.path.join(stories_dir, d, ".ready"))
            and not os.path.exists(os.path.join(stories_dir, d, ".private"))
            and not os.path.exists(os.path.join(stories_dir, d, ".published"))]


def _index(web_dir):
    out = []
    for name in sorted(os.listdir(web_dir), reverse=True):
        d = os.path.join(web_dir, name)
        if not os.path.isdir(d):
            continue
        e = {"id": name, "images": [], "photos": [], "portrait": None, "story": ""}
        for f in sorted(os.listdir(d)):
            if f == "story.txt":
                with open(os.path.join(d, f), encoding="utf-8", errors="ignore") as fh:
                    e["story"] = fh.read()
            elif f.startswith("storyteller_photo"):
                e["portrait"] = f
            elif f.lower().endswith((".png", ".jpg", ".jpeg")):
                (e["photos"] if "photo" in f else e["images"]).append(f)
        out.append(e)
    return out


def _publish_once(repo, stories_dir, names):
    """Reset the archive to the web, add the stories, push. Returns the number pushed."""
    web = os.path.join(repo, "docs", "stories")
    _git(["fetch", "-q", "origin", "main"], repo, 90, check=True)
    _git(["reset", "-q", "--hard", "origin/main"], repo, 30, check=True)
    _git(["clean", "-fdq", "docs"], repo, 30)
    os.makedirs(web, exist_ok=True)
    new = 0
    for name in names:
        src, dst = os.path.join(stories_dir, name), os.path.join(web, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.gcode", "*.wav", "*.json", ".private", ".published", ".ready"))
        new += 1
    with open(os.path.join(repo, "docs", "stories.json"), "w", encoding="utf-8") as f:
        json.dump(_index(web), f, ensure_ascii=False, indent=1)
    # -f: a stray "stories/" in .gitignore once silently kept every image out of the archive
    _git(["add", "-f", "docs"], repo, 30, check=True)
    r = _git(["commit", "-q", "-m", f"Publicar {new} historia(s)"], repo, 30)
    if "nothing to commit" in (r.stdout + r.stderr):
        return 0
    _git(["push", "-q", "origin", "HEAD:main"], repo, 180, check=True)
    return new


def sync(stories_dir=None):
    """Returns dict(status, new). status: off | mock | nothing | offline | published | error."""
    if mode() == "off":
        return {"status": "off", "new": 0}
    stories_dir = str(stories_dir or config.STORIES_DIR)
    names = pending(stories_dir)
    if mode() == "mock":
        return {"status": "mock", "new": len(names)}
    if not names:
        return {"status": "nothing", "new": 0}
    repo = str(config.PUBLISH_REPO_DIR)
    if not os.path.isdir(os.path.join(repo, ".git")):
        return {"status": "error", "new": 0, "error": "no hay archivo web (yachachiq-archivo)"}
    if not have_internet():
        return {"status": "offline", "new": len(names)}
    try:
        try:
            new = _publish_once(repo, stories_dir, names)
        except subprocess.CalledProcessError as e:
            # somebody else pushed in between (or a stale clone): start over once from the web
            log.info("publish: retrying after git error: %s", (e.stderr or "")[:200])
            new = _publish_once(repo, stories_dir, names)
        for name in names:
            with open(os.path.join(stories_dir, name, ".published"), "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        log.info("published %d new stories", new)
        return {"status": "published", "new": new}
    except Exception as e:
        err = getattr(e, "stderr", "") or str(e)
        log.warning("publish failed: %s", str(err)[:300])
        return {"status": "error", "new": len(names), "error": str(err)[:200]}


def sync_later(on_done=None):
    """Publish in a background thread; returns at once with what is queued.
    on_done(result) is called from that thread when it finishes."""
    n = len(pending()) if mode() != "off" else 0
    if mode() in ("off", "mock") or not n:
        return sync()

    def work():
        while True:
            _again["flag"] = False
            res = sync()
            if on_done:
                try:
                    on_done(res)
                except Exception:
                    pass
            if not _again["flag"]:
                break

    if _lock.acquire(blocking=False):
        def run():
            try:
                work()
            finally:
                _lock.release()
        threading.Thread(target=run, daemon=True).start()
    else:
        _again["flag"] = True                  # a sync is running: it will go round once more
    return {"status": "queued", "new": n}
