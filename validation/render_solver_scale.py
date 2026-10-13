#!/usr/bin/env python3
"""Render sampled outer-surface states from solver_scale_capture (real CSV only)."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


source = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
frames = {}
dt = 0.0
steps = 0
for row in csv.reader(source.open()):
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "F":
        frames[int(row[1])] = {"state": list(map(float, row[2:])), "voxels": []}
    elif row[0] == "V":
        frames[int(row[1])]["voxels"].append(list(map(float, row[2:])))

expected_surface = len(frames[min(frames)]["voxels"])
assert len(frames) >= 2
assert all(len(frame["voxels"]) == expected_surface for frame in frames.values())
assert all(math.isfinite(value) for frame in frames.values() for voxel in frame["voxels"] for value in voxel)

width, height = 1600, 1000
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = lambda size: ImageFont.truetype(font_path, size)
side = 0.01
white = (232, 239, 248)
muted = (151, 169, 190)
orange = (245, 157, 78)
blue = (64, 174, 232)

def iso(p):
    x, y, z = p
    return ((x + y) * 0.70710678, (x - y) * 0.40824829 - z * 0.81649658)

all_points = [voxel[1:4] for frame in frames.values() for voxel in frame["voxels"]]
projected = [iso(p) for p in all_points]
low = [min(p[i] for p in projected) for i in range(2)]
high = [max(p[i] for p in projected) for i in range(2)]
scale = min(1250 / (high[0] - low[0] + 0.06), 670 / (high[1] - low[1] + 0.06))
origin = (800 - (low[0] + high[0]) * 0.5 * scale, 550 - (low[1] + high[1]) * 0.5 * scale)

def project(p):
    u, v = iso(p)
    return origin[0] + u * scale, origin[1] + v * scale

def draw_cube(draw, voxel):
    _, x, y, z, damage, lost, _physical_index = voxel
    half = side * 0.5
    cx, cy, cz = x, y, z
    faces = [
        ([(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)], 0.72),
        ([(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)], 0.86),
        ([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], 1.12),
    ]
    if lost:
        base = (240, 75, 57)
    elif damage > 0.0:
        base = orange
    elif abs(cz - 0.005) < 1e-6:
        base = blue
    elif cz > 0.30:
        base = orange
    else:
        base = (130, 157, 178)
    for vertices, shade in faces:
        polygon = [project((cx + a * half, cy + b * half, cz + c * half)) for a, b, c in vertices]
        color = tuple(max(0, min(255, int(channel * shade))) for channel in base)
        draw.polygon(polygon, fill=color, outline=(35, 49, 64))

ordered = sorted(frames)
stats = []
initial = {int(v[0]): v[1:4] for v in frames[ordered[0]]["voxels"]}
for index, frame_id in enumerate(ordered):
    frame = frames[frame_id]
    state = frame["state"]
    image = Image.new("RGB", (width, height), (11, 18, 29))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((30, 30, 1570, 950), 20, fill=(15, 26, 40), outline=(43, 61, 79), width=2)
    voxels = frame["voxels"]
    voxels.sort(key=lambda voxel: voxel[1] - voxel[2] + voxel[3])
    for voxel in voxels:
        draw_cube(draw, voxel)
    max_delta = max(
        math.dist(v[1:4], initial[int(v[0])])
        for v in voxels
    )
    draw.text((58, 52), "36 864 REV · ГРАВИТАЦИЯ · ЗАКРЕПЛЁННЫЙ НИЖНИЙ СЛОЙ", font=font(27), fill=white)
    draw.text((60, 96), "Физический block-sparse PCG workload · поверхности реальной сцены", font=font(18), fill=muted)
    draw.text((60, 915), f"t = {frame_id * dt:.3f} s · кадр {frame_id}/{steps} · разорвано связей: {int(state[9])} · max Δ поверхности: {max_delta * 1000:.4f} мм", font=font(17), fill=muted)
    image.save(out_dir / f"{index:05}.png")
    stats.append({"frame": frame_id, "broken": int(state[9]), "clusters": int(state[10]), "max_surface_displacement_m": max_delta})
    if index % 30 == 0:
        print(f"render {index}/{len(ordered)}", flush=True)

Path(str(source) + ".render.json").write_text(json.dumps({"frames": len(ordered), "dt": dt, "steps": steps, "planned_seconds": steps * dt, "captured_seconds": (ordered[-1] - ordered[0]) * dt, "visible_surface_revs": expected_surface, "stats": stats}, indent=2))
sequence = str(out_dir / "%05d.png")
gif = source.with_suffix(".gif")
video = source.with_suffix(".mp4")
subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", "10", "-i", sequence,
                "-filter_complex", "[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer", str(gif)], check=True)
subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", "10", "-i", sequence,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(video)], check=True)
print(f"done {len(ordered)} frames: {gif} and {video}")
