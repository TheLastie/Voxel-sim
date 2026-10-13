#!/usr/bin/env python3
"""Render actual thermal_capture CSV states to 1600x1000 PNG frames."""
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

source = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
dt = 1.0
steps = 0
frames = {}
for row in csv.reader(source.open()):
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "S":
        frames[int(row[1])] = []
    elif row[0] == "Q":
        frames[int(row[1])].append((int(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])))

if not frames or any(len(values) != 24 for values in frames.values()):
    raise SystemExit("expected complete 24-REV thermal snapshots")

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = lambda size: ImageFont.truetype(font_path, size)
white, muted = (235, 242, 250), (153, 174, 198)

def heat_color(temp):
    amount = max(0.0, min(1.0, (temp - 300.0) / 150.0))
    stops = ((38, 112, 218), (250, 202, 71), (225, 54, 48))
    scaled = amount * 2
    index = min(1, int(scaled))
    fraction = scaled - index
    return tuple(round(stops[index][c] * (1 - fraction) + stops[index + 1][c] * fraction) for c in range(3))

for frame_index, (step, cells) in enumerate(sorted(frames.items())):
    temperatures = [cell[4] for cell in cells]
    image = Image.new("RGB", (1600, 1000), (11, 18, 29))
    draw = ImageDraw.Draw(image)
    draw.text((54, 32), "ТЕПЛОПРОВОДНОСТЬ · ЖЕЛЕЗНЫЙ СТЕРЖЕНЬ · 24 REV", font=font(30), fill=white)
    draw.text((56, 82), "Физические состояния движка · 1 REV = 1 см³ · цвет показывает температуру", font=font(20), fill=muted)
    draw.rounded_rectangle((36, 136, 1564, 940), 24, fill=(16, 26, 40), outline=(43, 61, 79), width=2)
    draw.text((76, 164), f"Время расчёта: {step * dt:.0f} с    Кадр: {frame_index + 1}/{len(frames)}", font=font(24), fill=white)

    graph = (120, 255, 1470, 515)
    draw.rounded_rectangle(graph, 12, fill=(9, 16, 28), outline=(55, 75, 96), width=2)
    for temp in (300, 350, 400, 450):
        y = graph[3] - (temp - 300) / 150 * (graph[3] - graph[1])
        draw.line((graph[0], y, graph[2], y), fill=(42, 57, 75), width=1)
        draw.text((graph[0] - 64, y - 12), str(temp), font=font(16), fill=muted)
    points = []
    for index, temp in enumerate(temperatures):
        x = graph[0] + index * (graph[2] - graph[0]) / (len(temperatures) - 1)
        y = graph[3] - (temp - 300) / 150 * (graph[3] - graph[1])
        points.append((x, y))
    draw.line(points, fill=(248, 190, 74), width=5)
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 226, 147))
    draw.text((122, 526), "Профиль вдоль стержня · K", font=font(18), fill=muted)

    start_x, base_y, cell_w, gap, height = 86, 746, 55, 7, 120
    for index, (_, _, _, _, temp) in enumerate(cells):
        x = start_x + index * (cell_w + gap)
        y_top = base_y - height
        color = heat_color(temp)
        top = tuple(min(255, round(c * 1.2)) for c in color)
        side = tuple(round(c * 0.68) for c in color)
        draw.polygon(((x, y_top), (x + 14, y_top - 14), (x + cell_w + 14, y_top - 14), (x + cell_w, y_top)), fill=top, outline=(205, 215, 226))
        draw.polygon(((x + cell_w, y_top), (x + cell_w + 14, y_top - 14), (x + cell_w + 14, base_y - 14), (x + cell_w, base_y)), fill=side, outline=(205, 215, 226))
        draw.rectangle((x, y_top, x + cell_w, base_y), fill=color, outline=(205, 215, 226), width=2)
        if index % 3 == 0:
            draw.text((x + 2, base_y + 12), f"{temp:.0f} K", font=font(14), fill=white)
    draw.text((84, 866), "Слева — начальная горячая половина (450 K); справа — холодная (300 K). Кубики остаются на решётке.", font=font(18), fill=muted)
    image.save(output / f"{frame_index:05}.png")
    if frame_index % 32 == 0:
        print(f"render {frame_index + 1}/{len(frames)}", flush=True)
print(f"rendered {len(frames)} physical states, duration={steps * dt:g} s", flush=True)
