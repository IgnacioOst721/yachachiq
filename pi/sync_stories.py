# sync_stories.py -- "store-and-forward" uploader
# The plotter always saves stories locally. When ANY internet connection is
# available (phone hotspot, town wifi, satellite), run this script to publish
# every pending story to the public web gallery (GitHub Pages).
#
# Setup (once, on the machine that has the stories/ folder):
#   git clone https://github.com/IgnacioOst721/yachachiq.git ~/yachachiq
#   (log in once with: gh auth login   or configure a git token)
#
# Usage:
#   python3 sync_stories.py               # syncs ./stories -> repo docs/stories
#   python3 sync_stories.py /path/to/stories

import json
import os
import shutil
import subprocess
import sys

REPO_DIR = os.path.expanduser("~/yachachiq")
WEB_STORIES = os.path.join(REPO_DIR, "docs", "stories")


def have_internet():
    try:
        subprocess.run(["ping", "-c", "1", "-W", "3", "github.com"],
                       capture_output=True, timeout=10, check=True)
        return True
    except Exception:
        return False


def sync(stories_dir):
    if not os.path.isdir(stories_dir):
        print(f"No existe la carpeta {stories_dir}")
        return
    if not have_internet():
        print("Sin conexión. Las historias quedan guardadas localmente;")
        print("vuelve a correr este script cuando haya internet.")
        return

    os.makedirs(WEB_STORIES, exist_ok=True)
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--rebase"], check=False)

    new = 0
    for name in sorted(os.listdir(stories_dir)):
        src = os.path.join(stories_dir, name)
        dst = os.path.join(WEB_STORIES, name)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
            new += 1
            print(f"  + {name}")

    # rebuild the index the web page reads
    index = []
    for name in sorted(os.listdir(WEB_STORIES), reverse=True):
        d = os.path.join(WEB_STORIES, name)
        if not os.path.isdir(d):
            continue
        entry = {"id": name, "images": [], "photos": [], "story": ""}
        for f in sorted(os.listdir(d)):
            if f == "story.txt":
                with open(os.path.join(d, f), encoding="utf-8", errors="ignore") as fh:
                    entry["story"] = fh.read()
            elif f.endswith((".png", ".jpg")):
                if "photo" in f:
                    entry["photos"].append(f)
                else:
                    entry["images"].append(f)
        index.append(entry)
    with open(os.path.join(REPO_DIR, "docs", "stories.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    if new == 0:
        print("No hay historias nuevas que subir.")

    subprocess.run(["git", "-C", REPO_DIR, "add", "docs"], check=True)
    r = subprocess.run(["git", "-C", REPO_DIR, "commit",
                        "-m", f"Publicar {new} historia(s) nueva(s)"],
                       capture_output=True, text=True)
    if "nothing to commit" in (r.stdout + r.stderr):
        print("Nada nuevo que publicar.")
        return
    subprocess.run(["git", "-C", REPO_DIR, "push"], check=True)
    print(f"✅ {new} historia(s) publicadas en el archivo web.")


if __name__ == "__main__":
    sync(sys.argv[1] if len(sys.argv) > 1 else "stories")
