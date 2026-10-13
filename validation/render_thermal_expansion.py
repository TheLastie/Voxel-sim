#!/usr/bin/env python3
"""Render actual restrained thermal-expansion states from the engine CSV."""
import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/thermal-expansion.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("validation/thermal-expansion")
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

if len(frames) < 176 or any(frame["metrics"] is None or len(frame["voxels"]) != 3 for frame in frames.values()):
    raise SystemExit(f"incomplete capture: {len(frames)} states with three voxels and metrics required")
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
    target, t0, t1, t2, heater, free_span, actual_span, stress, clusters, force = frame["metrics"]
    series.append((step, (t0 + t1 + t2) / 3.0, free_span, actual_span, stress / 1e6))
max_stress = max(200.0, max(item[4] for item in series) * 1.1)

def line_chart(draw, values, box, color, low, high, label):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(49, 72, 96), width=1)
    points = []
    for step, value in values:
        x = x0 + (x1 - x0) * step / steps
        y = y1 - (y1 - y0) * (value - low) / max(high - low, 1e-12)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=color, width=3)
    draw.text((x0 + 8, y0 + 5), label, font=font(14), fill=color)

images = []
prefix.parent.mkdir(parents=True, exist_ok=True)
for frame_no, (step, frame) in enumerate(sorted(frames.items())):
    target, t0, t1, t2, heater, free_span, actual_span, stress, clusters, force = frame["metrics"]
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw.text((34, 22), "RESTRAINED THERMAL EXPANSION · ENGINE STATE", font=font(23), fill=white)
    draw.text((36, 60), f"t = {step * dt:.2f} s   frame {frame_no + 1}/{len(frames)}   solver step {step}/{steps}", font=font(16), fill=muted)

    # Horizontal rod layout: endpoint positions and the middle REV come directly from CSV.
    rod_box = (70, 118, 690, 300)
    draw.rounded_rectangle(rod_box, radius=16, fill=(12, 27, 43), outline=(49, 72, 96), width=2)
    xs = [cell[1] for cell in sorted(frame["voxels"])]
    center_y = 205
    projected = [150 + (x - xs[0]) / 0.02 * 420 for x in xs]
    draw.line((projected[0], center_y, projected[-1], center_y), fill=(125, 154, 183), width=5)
    for i, (_, x, _, _, temp) in enumerate(sorted(frame["voxels"])):
        color_mix = max(0.0, min(1.0, (temp - 293.15) / 70.0))
        color = (int(72 + 175 * color_mix), int(143 - 49 * color_mix), int(225 - 142 * color_mix))
        draw.rectangle((projected[i] - 24, center_y - 24, projected[i] + 24, center_y + 24), fill=color, outline=white, width=2)
        draw.text((projected[i] - 24, center_y + 34), f"{temp:.1f} K", font=font(12), fill=muted)
        if i in (0, 2):
            draw.line((projected[i], center_y - 42, projected[i], center_y + 42), fill=(252, 198, 90), width=5)
    draw.text((92, 270), f"реальный пролёт закреплённых концов: {actual_span * 1000:.6f} mm", font=font(15), fill=white)
    draw.text((92, 290), f"свободный пролёт по αΔT: {free_span * 1000:.6f} mm", font=font(15), fill=(252, 198, 90))

    stat_box = (730, 118, 1088, 300)
    draw.rounded_rectangle(stat_box, radius=16, fill=(12, 27, 43), outline=(49, 72, 96), width=2)
    stats = [f"T узлов: {t0:.2f}, {t1:.2f}, {t2:.2f} K",
             f"реакция связи: {stress / 1e6:.3f} MPa",
             f"реакция силы: {force:.2f} N",
             f"энергия нагревателя: {heater:.4f} J",
             f"целых кластеров: {int(clusters)}"]
    for n, text in enumerate(stats):
        draw.text((750, 145 + 29 * n), text, font=font(14), fill=muted)

    temps = [(s, temp) for s, temp, _, _, _ in series if s <= step]
    stresses = [(s, value) for s, _, _, _, value in series if s <= step]
    lengths = [(s, 1000 * value) for s, _, _, value, _ in series if s <= step]
    line_chart(draw, temps, (70, 340, 520, 625), (111, 182, 255), 285, 375, "Температура (K)")
    line_chart(draw, stresses, (570, 340, 1020, 470), (255, 158, 94), 0, max_stress, "Сжимающее напряжение (MPa)")
    line_chart(draw, lengths, (570, 500, 1020, 625), (117, 220, 163), 19.995, 20.03, "Фактический пролёт (mm)")
    draw.text((35, 664), "Координаты и температура — из сохранённых engine states; расчётная свободная длина — справочная.", font=font(13), fill=muted)
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
