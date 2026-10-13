#!/usr/bin/env python3
"""Render actual engine states comparing fiber-parallel and transverse wood."""
import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/thermal-expansion-anisotropy.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("validation/thermal-expansion-anisotropy")
dt, steps = 0.01, 0
frames = {}
for row in csv.reader(source.open(newline="")):
    if not row:
        continue
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "S":
        frames[int(row[1])] = {"voxels": [], "metrics": None}
    elif row[0] == "Q":
        frames[int(row[1])]["voxels"].append(
            (int(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]))
        )
    elif row[0] == "X":
        frames[int(row[1])]["metrics"] = tuple(map(float, row[2:]))

if len(frames) < 176 or any(frame["metrics"] is None or len(frame["voxels"]) != 6 for frame in frames.values()):
    raise SystemExit(f"incomplete capture: {len(frames)} states with six voxels and metrics required")
if max(frames) * dt < 17.6 - 1e-9:
    raise SystemExit("physical trace is shorter than 17.6 seconds")

width, height = 1120, 700
background, white, muted = (7, 16, 28), (235, 242, 250), (153, 174, 198)
try:
    def font(size):
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
except OSError:
    font = ImageFont.load_default

series = []
for step, frame in sorted(frames.items()):
    m = frame["metrics"]
    series.append((step, m[6] * 1000, m[7] * 1000, m[8] * 1000, m[9] * 1000, m[10] / 1e6, m[11] / 1e6))
stress_max = max(3.5, max(max(item[5], item[6]) for item in series) * 1.1)

def chart(draw, box, values, low, high, lines, title):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(49, 72, 96), width=1)
    draw.text((x0 + 8, y0 + 5), title, font=font(13), fill=white)
    for line_number, (value_index, color, label) in enumerate(lines):
        legend_x = x0 + 8 + (line_number % 2) * ((x1 - x0) // 2)
        legend_y = y0 + 22 + (line_number // 2) * 14
        draw.text((legend_x, legend_y), label, font=font(10), fill=color)
        points = []
        for item in values:
            x = x0 + (x1 - x0) * item[0] / steps
            y = y1 - 8 - (y1 - y0 - 55) * (item[value_index] - low) / max(high - low, 1e-12)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)

images = []
prefix.parent.mkdir(parents=True, exist_ok=True)
for frame_no, (step, frame) in enumerate(sorted(frames.items())):
    m = frame["metrics"]
    target, t0, t1, t3, t4, heater, free_par, free_perp, actual_par, actual_perp, stress_par, stress_perp, clusters, force_par = m
    voxels = sorted(frame["voxels"])
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw.text((34, 22), "ANISOTROPIC THERMAL EXPANSION · WOOD", font=font(23), fill=white)
    draw.text((36, 60), f"t = {step * dt:.2f} s   frame {frame_no + 1}/{len(frames)}   solver step {step}/{steps}", font=font(16), fill=muted)

    draw.rounded_rectangle((55, 112, 1065, 360), radius=16, fill=(12, 27, 43), outline=(49, 72, 96), width=2)
    draw.text((80, 126), "Поперёк волокон · направление X · α⊥", font=font(16), fill=(255, 180, 102))
    draw.text((80, 240), "Вдоль волокон · направление Z · α‖", font=font(16), fill=(119, 220, 166))
    y_rows = [180, 294]
    for group, coord_axis, y in ((voxels[:3], 1, y_rows[0]), (voxels[3:], 3, y_rows[1])):
        coords = [cell[coord_axis] for cell in group]
        for cell, coord in zip(group, coords):
            x = 405 + (coord - coords[0]) / 0.02 * 440
            color_mix = max(0.0, min(1.0, (cell[4] - 293.15) / 70.0))
            color = (int(72 + 175 * color_mix), int(143 - 49 * color_mix), int(225 - 142 * color_mix))
            draw.rectangle((x - 21, y - 21, x + 21, y + 21), fill=color, outline=white, width=2)
            if cell[0] in (0, 2, 3, 5):
                draw.line((x, y - 30, x, y + 30), fill=(252, 198, 90), width=4)
    draw.text((405, 207), f"пролёт {actual_perp * 1000:.6f} mm · свободный {free_perp * 1000:.6f} mm", font=font(13), fill=(255, 180, 102))
    draw.text((405, 321), f"пролёт {actual_par * 1000:.6f} mm · свободный {free_par * 1000:.6f} mm", font=font(13), fill=(119, 220, 166))

    draw.rounded_rectangle((55, 370, 1065, 431), radius=12, fill=(12, 27, 43), outline=(49, 72, 96), width=1)
    draw.text((76, 378), f"T: {t0:.2f}, {t1:.2f}, {t3:.2f}, {t4:.2f} K", font=font(13), fill=muted)
    draw.text((420, 378), f"сжатие вдоль: {stress_par / 1e6:.3f} MPa", font=font(13), fill=(119, 220, 166))
    draw.text((730, 378), f"поперёк: {stress_perp / 1e6:.3f} MPa", font=font(13), fill=(255, 180, 102))
    draw.text((76, 404), f"энергия нагревателя: {heater:.3f} J · целые кластеры: {int(clusters)}", font=font(12), fill=muted)

    sofar = [item for item in series if item[0] <= step]
    chart(draw, (55, 445, 530, 625), sofar, 19.995, 20.055,
          [(1, (119, 220, 166), "‖ свободно"), (2, (255, 180, 102), "⊥ свободно"),
           (3, (89, 151, 204), "‖ закреплено"), (4, (161, 121, 200), "⊥ закреплено")], "Пролёт (mm)")
    chart(draw, (565, 445, 1065, 625), sofar, 0, stress_max,
          [(5, (119, 220, 166), "‖ волокнам"), (6, (255, 180, 102), "⊥ волокнам")], "Реакционное напряжение (MPa)")
    draw.text((35, 664), "Позиции/температуры/реакции взяты из сохранённого engine trace; свободные длины — сравнение αΔT.", font=font(13), fill=muted)
    images.append(image)
    if frame_no % 32 == 0:
        print(f"render {frame_no + 1}/{len(frames)}", flush=True)

gif_path, mp4_path = prefix.with_suffix(".gif"), prefix.with_suffix(".mp4")
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
print(f"rendered {len(frames)} actual states over {max(frames) * dt:.1f}s: {gif_path}, {mp4_path}")
