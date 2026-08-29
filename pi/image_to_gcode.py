# GRBL-compatible: G0 = travel (pen up), G1 = draw (pen down), feed rate
# included.

import cv2
import numpy as np
import sys
from pathlib import Path

# === ADJUST THESE FOR YOUR IMAGE ===
THRESHOLD = 100
MIN_CONTOUR_LENGTH = 25
SIMPLIFY = 0.0015
DILATE = True
DILATE_ITER = 1

# === PLOTTER (set to YOUR machine's drawing area in mm) ===
MAX_X = 50.0
MAX_Y = 40.0
MARGIN = 0.05
FEED_RATE = 1500   # mm/min drawing speed -- REQUIRED by GRBL, tune to taste


def process_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    if DILATE:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=DILATE_ITER)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.arcLength(c, False) > MIN_CONTOUR_LENGTH]
    contours.sort(key=lambda c: cv2.arcLength(c, False), reverse=True)

    scale_x = MAX_X / w
    scale_y = MAX_Y / h
    scale = min(scale_x, scale_y) * (1 - MARGIN)

    drawn_w = w * scale
    drawn_h = h * scale
    offset_x = (MAX_X - drawn_w) / 2
    offset_y = (MAX_Y - drawn_h) / 2

    return img, binary, contours, scale, offset_x, offset_y, w, h


def make_gcode(contours, scale, offset_x, offset_y, w, h, source):
    lines = [
        f"; Generated from {Path(source).name}",
        f"; {len(contours)} contours",
        "G21 ; mm",
        "G90 ; absolute",
        "M5  ; pen up",
        "",
    ]

    total_pts = 0
    paths = []

    for contour in contours:
        epsilon = SIMPLIFY * cv2.arcLength(contour, False)
        simplified = cv2.approxPolyDP(contour, epsilon, False)
        if len(simplified) < 2:
            continue

        path_pts = []
        px, py = simplified[0][0]
        x_mm = offset_x + px * scale
        y_mm = MAX_Y - (offset_y + py * scale)
        lines.append(f"G0 X{x_mm:.2f} Y{y_mm:.2f}")
        path_pts.append((x_mm, y_mm, False))

        for pt in simplified[1:]:
            px, py = pt[0]
            x_mm = offset_x + px * scale
            y_mm = MAX_Y - (offset_y + py * scale)
            lines.append(f"G1 X{x_mm:.2f} Y{y_mm:.2f} F{FEED_RATE}")
            path_pts.append((x_mm, y_mm, True))
            total_pts += 1

        paths.append(path_pts)

    lines.append("")
    lines.append("M2")

    return lines, paths, total_pts


def convert(image_path, output_path):
    """Headless: image file -> gcode file. Safe to call on the Pi (no GUI)."""
    img, binary, contours, scale, ox, oy, w, h = process_image(image_path)
    lines, paths, total_pts = make_gcode(contours, scale, ox, oy, w, h, image_path)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  gcode: {len(contours)} contours, {total_pts} points -> {output_path}")
    return total_pts


def show_preview(img, binary, contours, paths, total_pts):
    import matplotlib.pyplot as plt   # imported here so the Pi never needs matplotlib
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Original"); axes[0].axis("off")

    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title(f"2. Threshold ({THRESHOLD})"); axes[1].axis("off")

    contour_img = np.ones_like(img) * 255
    cv2.drawContours(contour_img, contours, -1, (0, 0, 0), 1)
    axes[2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"3. {len(contours)} contours"); axes[2].axis("off")

    ax = axes[3]
    ax.plot([0, MAX_X, MAX_X, 0, 0], [0, 0, MAX_Y, MAX_Y, 0], "r--", linewidth=0.5)
    for path in paths:
        last = None
        for x, y, drawing in path:
            if last and drawing:
                ax.plot([last[0], x], [last[1], y], "k-", linewidth=0.7)
            last = (x, y)
    ax.set_xlim(-5, MAX_X + 5); ax.set_ylim(-5, MAX_Y + 5)
    ax.set_aspect("equal")
    ax.set_title(f"4. Plotter preview\n{total_pts} points")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 image_to_gcode.py image.png out.gcode   # make gcode (no window)")
        print("  python3 image_to_gcode.py image.png             # preview only (needs a screen)")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Processing {image_path} ...")

    if output_path:
        convert(image_path, output_path)        # headless -> safe on the Pi
    else:
        img, binary, contours, scale, ox, oy, w, h = process_image(image_path)
        lines, paths, total_pts = make_gcode(contours, scale, ox, oy, w, h, image_path)
        print(f"Found {len(contours)} contours, {total_pts} points")
        try:
            show_preview(img, binary, contours, paths, total_pts)
        except Exception as e:
            print("(preview unavailable:", e, ")")


if __name__ == "__main__":
    main()
