#!/usr/bin/env python3
"""Render physical thermal or melting CSV snapshots as an isometric 3D scene."""
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

source, output = Path(sys.argv[1]), Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
dt, steps = 1.0, 0
frames = {}
for row in csv.reader(source.open()):
    if not row:
        continue
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "S":
        frames[int(row[1])] = {"cells": [], "bonds": []}
    elif row[0] == "Q":
        frames[int(row[1])]["cells"].append(
            (int(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]) if len(row) > 7 else 0.0)
        )
    elif row[0] == "B":
        frames[int(row[1])]["bonds"].append((int(row[2]), int(row[3])))

if not frames or len({len(frame["cells"]) for frame in frames.values()}) != 1:
    raise SystemExit("expected complete, consistently sized physical snapshots")

all_cells = [cell for frame in frames.values() for cell in frame["cells"]]
min_temp = min(cell[4] for cell in all_cells)
max_temp = max(cell[4] for cell in all_cells)
if max_temp - min_temp < 1:
    max_temp = min_temp + 1
coords = [(cell[1], cell[2], cell[3]) for cell in all_cells]
lo = [min(p[i] for p in coords) - 0.006 for i in range(3)]
hi = [max(p[i] for p in coords) + 0.006 for i in range(3)]
center = [(lo[i] + hi[i]) / 2 for i in range(3)]
extent = [(hi[i] - lo[i]) / 2 for i in range(3)]
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = lambda size: ImageFont.truetype(font_path, size)
white, muted = (235, 242, 250), (153, 174, 198)

def project(p):
    x, y, z = [p[i] - center[i] for i in range(3)]
    return (800 + (x + y) * .70710678 * scale, 555 + ((x - y) * .40824829 - z * .81649658) * scale)

def heat_color(temp):
    amount = max(0.0, min(1.0, (temp - min_temp) / (max_temp - min_temp)))
    stops = ((38, 112, 218), (250, 202, 71), (225, 54, 48))
    scaled = amount * 2
    index = min(1, int(scaled))
    fraction = scaled - index
    return tuple(round(stops[index][c] * (1 - fraction) + stops[index + 1][c] * fraction) for c in range(3))

padding_x, padding_y = 680, 430
screen_x = (extent[0] + extent[1]) * .70710678
screen_y = (extent[0] + extent[1]) * .40824829 + extent[2] * .81649658
scale = .78 * min(padding_x / max(screen_x, .005), padding_y / max(screen_y, .005))
cube = .0097
corners = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
faces = [((0,1,5,4),.62),((1,2,6,5),.78),((4,5,6,7),1.16)]

for frame_index, (step, frame) in enumerate(sorted(frames.items())):
    cells = sorted(frame["cells"])
    by_id = {cell[0]: cell for cell in cells}
    image = Image.new("RGB", (1600, 1000), (7, 16, 28))
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "ФИЗИЧЕСКОЕ СОСТОЯНИЕ · 3D REV", font=font(30), fill=white)
    liquid = max(cell[5] for cell in cells)
    draw.text((50, 76), f"Время: {step * dt:.3f} с · кадр {frame_index + 1}/{len(frames)} · связи: {len(frame['bonds'])} · жидкая доля: {liquid:.2f}", font=font(20), fill=muted)
    draw.rounded_rectangle((28, 124, 1572, 950), 20, fill=(11, 24, 39), outline=(43, 61, 79), width=2)
    # Draw the actual intact topology before the REV cubes.
    for i, j in frame["bonds"]:
        if i in by_id and j in by_id:
            a, b = project(by_id[i][1:4]), project(by_id[j][1:4])
            draw.line((a, b), fill=(151, 177, 201), width=7)
    ordered = sorted(cells, key=lambda cell: cell[1] - cell[2] + cell[3])
    for cell in ordered:
        _, x, y, z, temp, liquid = cell
        points = [project((x + dx*cube/2, y + dy*cube/2, z + dz*cube/2)) for dx,dy,dz in corners]
        color = heat_color(temp)
        for indices, shade in faces:
            poly = [points[i] for i in indices]
            face_color = tuple(round(value * shade) for value in color)
            draw.polygon(poly, fill=face_color, outline=(12, 25, 39))
    draw.text((52, 910), f"Температура: {min_temp:.0f}–{max_temp:.0f} K · жидкая фаза и целые связи взяты из состояния движка", font=font(18), fill=muted)
    image.save(output / f"{frame_index:05}.png")
    if frame_index % 32 == 0:
        print(f"render {frame_index + 1}/{len(frames)}", flush=True)
print(f"rendered {len(frames)} physical 3D states, duration={steps * dt:g} s", flush=True)
