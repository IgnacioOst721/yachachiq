# comfy_server.py  --  RUNS ON THE MAC
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, urllib.request, urllib.parse, uuid, time, random, re

COMFYUI_URL = "http://127.0.0.1:8188"

POSITIVE_SUFFIX = (
    ", simple black and white line drawing, coloring book page, "
    "clean bold outline, white background, no shading, no fill, "
    "no color, 2D flat illustration, thick contour lines, minimal detail, "
    "single outline only, children's coloring book, simple shapes, "
    "cartoon style, vector art, full body, clear subject"
)

NEGATIVE_PROMPT = (
    "photorealistic, realistic, shading, color, gradient, texture, "
    "shadow, 3d, detailed, photograph, painting, watercolor, noise, "
    "grayscale tones, dark background, blurry, bad quality, nudity, naked body, "
    "sketch lines, crosshatching, hatching, multiple lines, fur texture, "
    "detailed fur, messy lines, scribbles, rough sketch, abstract, swirls"
)

# ── prompt simplifier ─────────────────────────────────────────────────

def simplify_prompt(text):
    """Strip filler words, keep only visual nouns and actions."""
    filler = [
        "when i was", "i used to", "i remember", "once upon a time",
        "there was a", "there were", "i went to", "we went to",
        "a long time ago", "one day", "i think", "i feel", "i saw",
        "i would", "i could", "i should", "it was", "it is",
        "and then", "after that", "suddenly", "finally", "however"
    ]
    t = text.lower()
    for f in filler:
        t = t.replace(f, "")
    return t.strip(" ,.")

# ── scene extraction (fully offline) ─────────────────────────────────

def extract_scenes(story_text):
    sentences = re.split(r'[.!?]+', story_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    visual_words = [
        "walk", "run", "jump", "fly", "stand", "sit", "climb", "swim",
        "forest", "house", "mountain", "river", "sky", "tree", "flower",
        "dragon", "cat", "dog", "bird", "horse", "monster", "princess",
        "castle", "village", "ocean", "fire", "snow", "rain", "sun", "moon",
        "fight", "hug", "sleep", "eat", "play", "sing", "dance", "cry",
        "big", "small", "dark", "bright", "old", "young", "giant", "tiny",
        "girl", "boy", "man", "woman", "child", "baby", "king", "queen",
        "beach", "sea", "water", "sand", "wave", "boat", "ship", "island"
    ]

    def visual_score(sentence):
        s = sentence.lower()
        return sum(1 for w in visual_words if w in s)

    scored = sorted(enumerate(sentences), key=lambda x: visual_score(x[1]), reverse=True)
    top_indices = sorted([i for i, _ in scored[:5]])
    scenes = [sentences[i] for i in top_indices]

    if not scenes and sentences:
        scenes = sentences[:5]
    if not scenes:
        scenes = [story_text[:200]]

    return scenes

# ── image generation ──────────────────────────────────────────────────

def queue_prompt(prompt_text):
    simplified = simplify_prompt(prompt_text)
    print(f"  simplified prompt: {simplified!r}")
    workflow = {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": random.randint(0, 2**31),
            "steps": 25,
            "cfg": 11.0,
            "sampler_name": "euler",
            "scheduler": "karras",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "dreamshaper.safetensors"}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {
                  "text": simplified + POSITIVE_SUFFIX,
                  "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {
                  "text": NEGATIVE_PROMPT,
                  "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "pi_generated", "images": ["8", 0]}},
    }
    data = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["prompt_id"]

def wait_for_image(prompt_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as r:
            history = json.load(r)
        if prompt_id in history:
            entry = history[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error: {entry.get('status')}")
            for node_out in entry.get("outputs", {}).values():
                if node_out.get("images"):
                    return node_out["images"][0]
        time.sleep(1)
    raise TimeoutError("ComfyUI did not finish within %ds" % timeout)

def fetch_image_bytes(image_info):
    q = urllib.parse.urlencode({
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output")})
    with urllib.request.urlopen(f"{COMFYUI_URL}/view?{q}") as r:
        return r.read()

# ── request handler ───────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode())
        text = body.get("text", "").strip()
        mode = body.get("mode", "generate")

        print(f"\nReceived from Pi: mode={mode!r} text={text!r}")

        try:
            if mode == "extract_scenes":
                scenes = extract_scenes(text)
                print(f"  extracted {len(scenes)} scenes: {scenes}")
                resp_body = json.dumps({"scenes": scenes}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
            else:
                prompt_id = queue_prompt(text)
                print(f"  queued (prompt_id={prompt_id}); generating...")
                info = wait_for_image(prompt_id)
                png = fetch_image_bytes(info)
                print(f"  done -> sending {len(png)} bytes back to the Pi")
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)

        except Exception as e:
            print(f"  ERROR: {e}")
            msg = str(e).encode()
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def log_message(self, *args): pass

print("Server running on port 5001 ...")
HTTPServer(("0.0.0.0", 5001), Handler).serve_forever()
