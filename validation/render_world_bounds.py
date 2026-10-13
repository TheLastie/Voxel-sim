#!/usr/bin/env python3
"""Render actual world-boundary CSV states to a 3D GIF and MP4."""
import csv
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/world-bounds.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("validation/world-bounds")
rows = list(csv.reader(source.open(newline="")))
dt, steps = 0.01, 0
frames = {}
events = []
for row in rows:
    if not row:
        continue
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "S":
        frames[int(row[1])] = {"cells": [], "ledger": None}
    elif row[0] == "Q":
        frames[int(row[1])]["cells"].append(
            (int(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]))
        )
    elif row[0] == "L":
        frames[int(row[1])]["ledger"] = row
    elif row[0] == "E":
        events.append((int(row[1]), int(row[2]), row[3]))

if len(frames) < 176 or not frames or any(frame["ledger"] is None for frame in frames.values()):
    raise SystemExit(f"incomplete capture: {len(frames)} states, expected >=176 with ledger")
if max(frames) * dt < 17.6 - 1e-9:
    raise SystemExit(f"capture covers only {max(frames) * dt:.3f} simulated seconds")

try:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = lambda size: ImageFont.truetype(font_path, size)
except OSError:
    font = ImageFont.load_default
white, muted, blue = (235, 242, 250), (153, 174, 198), (111, 182, 255)
width, height = 1120, 700
center = (560, 390)
scale = 770.0

def project(p):
    x, y, z = p
    return (center[0] + (x + y) * .70710678 * scale,
            center[1] + ((x - y) * .40824829 - z * .81649658) * scale)

def draw_bounds(draw):
    lo, hi = -0.25, 0.25
    corners = [(x, y, z) for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)]
    pts = [project(p) for p in corners]
    edges = ((0, 2), (0, 4), (0, 1), (1, 3), (1, 5), (2, 3), (2, 6),
             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    # Subdued translucent-looking faces show the inclusive box without hiding REV.
    for face in ((0, 1, 3, 2), (0, 1, 5, 4), (0, 2, 6, 4)):
        draw.polygon([pts[i] for i in face], fill=(19, 39, 59))
    for a, b in edges:
        draw.line((pts[a], pts[b]), fill=(104, 133, 161), width=2)
    labels = [((-0.25, 0, 0), "X−"), ((0.25, 0, 0), "X+"), ((0, -0.25, 0), "Y−"),
              ((0, 0.25, 0), "Y+"), ((0, 0, -0.25), "Z−"), ((0, 0, 0.25), "Z+")]
    for position, label in labels:
        x, y = project(position)
        draw.text((x - 10, y - 10), label, font=font(15), fill=(178, 201, 222))

event_counts = {face: 0 for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
event_index = 0
images = []
prefix.parent.mkdir(parents=True, exist_ok=True)
for frame_number, (step, frame) in enumerate(sorted(frames.items())):
    while event_index < len(events) and events[event_index][0] <= step:
        event_counts[events[event_index][2]] += 1
        event_index += 1
    image = Image.new("RGB", (width, height), (7, 16, 28))
    draw = ImageDraw.Draw(image)
    draw.text((34, 22), "WORLD BOUNDS · ENGINE LEDGER · 3D", font=font(24), fill=white)
    draw.text((36, 60), f"t = {step * dt:.2f} s   кадр {frame_number + 1}/{len(frames)}   физический шаг {step}/{steps}", font=font(16), fill=muted)
    draw_bounds(draw)
    cells = sorted(frame["cells"], key=lambda cell: cell[1] - cell[2] + cell[3])
    cube_corners = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    cube_faces = ((0, 1, 5, 4), (1, 2, 6, 5), (4, 5, 6, 7))
    for _, x, y, z, temp in cells:
        amount = max(0.0, min(1.0, (temp - 280.0) / 360.0))
        color = (round(60 + 180 * amount), round(126 - 52 * amount), round(232 - 158 * amount))
        half = 0.006
        corners = [project((x + dx * half, y + dy * half, z + dz * half))
                   for dx, dy, dz in cube_corners]
        for face, shade in zip(cube_faces, (0.72, 0.84, 1.12)):
            face_color = tuple(max(0, min(255, round(channel * shade))) for channel in color)
            draw.polygon([corners[i] for i in face], fill=face_color, outline=(235, 242, 250))
    ledger = frame["ledger"]
    remaining, removed = int(ledger[2]), int(ledger[3])
    mass, internal, kinetic, pending = map(float, ledger[4:8])
    momentum = tuple(float(value) for value in ledger[8:11])
    panel = (748, 112, 1090, 590)
    draw.rounded_rectangle(panel, radius=14, fill=(12, 27, 43), outline=(49, 72, 96), width=2)
    draw.text((770, 132), "ГРАНИ / удалено", font=font(18), fill=white)
    for n, (face, title) in enumerate(zip(event_counts, ("X−", "X+", "Y−", "Y+", "Z−", "Z+"))):
        y = 174 + n * 31
        draw.text((774, y), f"{title}   {event_counts[face]} REV", font=font(16), fill=blue if event_counts[face] else muted)
    draw.line((770, 370, 1068, 370), fill=(49, 72, 96), width=1)
    stats = [f"осталось REV: {remaining}", f"удалено REV: {removed}", f"масса: {mass:.4e} кг",
             f"U ledger: {internal:.4e} Дж", f"K ledger: {kinetic:.4e} Дж",
             f"bond energy: {pending:.2e} Дж", f"p ledger: [{momentum[0]:.2e},",
             f"           {momentum[1]:.2e}, {momentum[2]:.2e}] кг·м/с"]
    for n, text in enumerate(stats):
        draw.text((774, 390 + n * 24), text, font=font(13), fill=muted)
    draw.text((35, 650), "Положение и цвет REV взяты из сохранённых состояний; ledger — накопленные данные движка.", font=font(14), fill=muted)
    images.append(image)
    if frame_number % 32 == 0:
        print(f"render {frame_number + 1}/{len(frames)}", flush=True)

gif_path = prefix.with_suffix(".gif")
mp4_path = prefix.with_suffix(".mp4")
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", "10", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path)]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
try:
    for image in images:
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed while writing MP4")
except BaseException:
    proc.kill()
    raise
print(f"rendered {len(frames)} actual states, simulated={max(frames) * dt:.1f}s, GIF={gif_path}, MP4={mp4_path}")
