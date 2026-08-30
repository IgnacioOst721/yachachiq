# listen.py  --  RUNS ON THE RASPBERRY PI (the "conductor")
# Records a 30-second story, extracts scenes, generates + prints in parallel.
# Stories are saved to a folder for safekeeping.

import subprocess, urllib.request, json, time, os, threading
from urllib.error import HTTPError
from datetime import datetime
import image_to_gcode, send_gcode

# ---- CONFIG (edit these) ----
MAC_IP      = "192.168.43.164"
MAC_PORT    = 5001
SERIAL_PORT = "/dev/ttyACM0"
BAUD        = 115200
AUDIO_FILE  = "speech.wav"
STORIES_DIR = "stories"   # folder where all stories are saved

# ── helpers ──────────────────────────────────────────────────────────

def record(filename=AUDIO_FILE, duration=30):
    print("\n🎙  Listening... tell your story! (30 seconds)")
    subprocess.run([
        "arecord", "-D", "plughw:2,0",
        "-f", "S16_LE", "-r", "16000", "-c", "1",
        "-d", str(duration), filename
    ])
    print("  Recording done.")

def transcribe(filename=AUDIO_FILE):
    print("  Transcribing...")
    r = subprocess.run(
        ["./build/bin/whisper-cli", "-m", "models/ggml-tiny.en.bin",
         "-f", filename, "--no-timestamps"],
        capture_output=True, text=True
    )
    text = r.stdout.strip()
    if "[BLANK_AUDIO]" in text or "(silence)" in text.lower():
        return ""
    return text

def request_image(scene_description, out_path):
    """Send one scene to the Mac, save PNG to out_path."""
    data = json.dumps({"text": scene_description}).encode()
    req = urllib.request.Request(
        f"http://{MAC_IP}:{MAC_PORT}",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            img = resp.read()
    except HTTPError as e:
        raise RuntimeError("Mac error: " + e.read().decode(errors="ignore"))
    with open(out_path, "wb") as f:
        f.write(img)
    print(f"  📸 Image saved: {out_path} ({len(img)} bytes)")
    return out_path

def request_scenes(story_text):
    """Ask the Mac server to extract scenes from the full story."""
    print("\n  📖 Extracting scenes from story...")
    data = json.dumps({"text": story_text, "mode": "extract_scenes"}).encode()
    req = urllib.request.Request(
        f"http://{MAC_IP}:{MAC_PORT}",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.load(resp)
            return result.get("scenes", [])
    except Exception as e:
        raise RuntimeError(f"Scene extraction error: {e}")

def save_story(story_dir, story_text, scenes):
    """Save story text and scene list to a text file."""
    story_file = os.path.join(story_dir, "story.txt")
    with open(story_file, "w") as f:
        f.write("STORY\n")
        f.write("=" * 40 + "\n")
        f.write(story_text + "\n\n")
        f.write("SCENES\n")
        f.write("=" * 40 + "\n")
        for i, scene in enumerate(scenes, 1):
            f.write(f"{i}. {scene}\n")
    print(f"  💾 Story saved to {story_file}")

# ── main pipeline ─────────────────────────────────────────────────────

def run_story(story_text, story_dir):
    """
    Extract scenes, then pipeline:
      - Thread A generates image N+1 while
      - Main thread prints image N
    """
    scenes = request_scenes(story_text)
    if not scenes:
        print("  No scenes extracted. Try again.")
        return

    print(f"\n  Found {len(scenes)} scenes:")
    for i, s in enumerate(scenes, 1):
        print(f"    {i}. {s}")

    # Save story text + scenes to folder
    save_story(story_dir, story_text, scenes)

    # Pre-generate first image before starting the loop
    img_paths  = [None] * len(scenes)
    gcode_paths = [None] * len(scenes)

    def generate(i):
        """Generate image and convert to gcode for scene i (0-indexed)."""
        img_path   = os.path.join(story_dir, f"scene_{i+1}.png")
        gcode_path = os.path.join(story_dir, f"scene_{i+1}.gcode")
        try:
            print(f"\n  🎨 Generating scene {i+1}: {scenes[i]!r}")
            request_image(scenes[i], img_path)
            image_to_gcode.convert(img_path, gcode_path)
            img_paths[i]   = img_path
            gcode_paths[i] = gcode_path
            print(f"  ✅ Scene {i+1} ready to print.")
        except Exception as e:
            print(f"  ❌ Scene {i+1} generation failed: {e}")

    # Start generating scene 1 immediately
    next_thread = threading.Thread(target=generate, args=(0,))
    next_thread.start()

    for i in range(len(scenes)):
        print(f"\n── Scene {i+1}/{len(scenes)} ──────────────────")

        # Wait for current scene to be ready
        next_thread.join()

        # Start generating NEXT scene in background while we print current
        if i + 1 < len(scenes):
            next_thread = threading.Thread(target=generate, args=(i+1,))
            next_thread.start()

        # Print current scene
        if gcode_paths[i]:
            if i > 0:
                input(f"\n  Place paper for scene {i+1}, then press ENTER...")
            send_gcode.send(gcode_paths[i], SERIAL_PORT, BAUD)
            print(f"  🖨  Scene {i+1} printed!")
            # automatic photo of the finished drawing (camera mounted over the bed)
            try:
                import capture_drawing
                capture_drawing.capture(os.path.join(story_dir, f"scene_{i+1}_photo.png"))
            except Exception as e:
                print(f"  (foto del dibujo no tomada: {e})")
        else:
            print(f"  ⚠️  Skipping scene {i+1} (generation failed).")

    print(f"\n🎉 Picture book complete! Saved in: {story_dir}")

    # try to publish to the web archive automatically (store-and-forward:
    # if there is no internet right now, the story stays queued locally and
    # will be uploaded the next time any connection is available)
    try:
        import sync_stories
        threading.Thread(target=sync_stories.sync, args=(STORIES_DIR,),
                         daemon=True).start()
    except Exception as e:
        print(f"  (publicación web pendiente: {e})")

# ── main loop ─────────────────────────────────────────────────────────

os.makedirs(STORIES_DIR, exist_ok=True)

print("=" * 50)
print("  YACHACHIQ STORY PLOTTER")
print("  Tell a story. It will become a picture book.")
print("=" * 50)

while True:
    input("\nPress ENTER when ready to record your story...")
    record()
    story = transcribe()
    print(f"\n📝 Story: {story!r}")

    if not story.strip():
        print("  No speech detected. Try again.")
        time.sleep(1)
        continue

    # Create a timestamped folder for this story
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    story_dir  = os.path.join(STORIES_DIR, timestamp)
    os.makedirs(story_dir, exist_ok=True)
    print(f"  📁 Story folder: {story_dir}")

    try:
        run_story(story, story_dir)
    except Exception as e:
        print(f"  Pipeline error: {e}")

    time.sleep(1)
