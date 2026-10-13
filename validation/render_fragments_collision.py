#!/usr/bin/env python3
"""Render saved fatigue-fracture and fragment-collision physical states."""
import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/fatigue-fragment-collision.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("validation/fatigue-fragment-collision")
dt, steps, frames, events, heat_events = 0.001, 0, {}, [], []
for row in csv.reader(source.open(newline="")):
    if not row:
        continue
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "S":
        frames[int(row[1])] = {"voxels": [], "bonds": [], "ledger": None}
    elif row[0] == "Q":
        frames[int(row[1])]["voxels"].append(
            (int(row[2]), *map(float, row[3:7]), 0.0)
        )
    elif row[0] == "V":
        state = frames[int(row[1])]["voxels"]
        voxel_id = int(row[2])
        prior = state[voxel_id]
        state[voxel_id] = (*prior[:5], float(row[6]))
    elif row[0] == "B":
        frames[int(row[1])]["bonds"].append((int(row[2]), int(row[3])))
    elif row[0] == "L":
        frames[int(row[1])]["ledger"] = row
    elif row[0] == "E":
        events.append((int(row[1]), row[3]))
        if row[3].startswith("fragment_collision_heat_"):
            heat_events.append((int(row[1]), float(row[3].removeprefix("fragment_collision_heat_").removesuffix("_J"))))

if len(frames) < 176 or max(frames, default=0) * dt < 17.6 - 1e-9:
    raise SystemExit(f"capture incomplete: {len(frames)} states over {max(frames, default=0)*dt:.3f} s")
if any(frame["ledger"] is None for frame in frames.values()):
    raise SystemExit("every rendered frame must include a conservation ledger")

W, H = 1600, 1000
scale = 12_000.0
palette = [(80, 178, 246), (243, 166, 80), (139, 216, 141)]
white, muted = (235, 242, 250), (153, 174, 198)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = lambda size: ImageFont.truetype(font_path, size)
prefix.parent.mkdir(parents=True, exist_ok=True)
images, events_seen, first_heat = [], set(), 0.0
cube = [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5),
        (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]
faces = ((0, 1, 5, 4), (1, 2, 6, 5), (4, 5, 6, 7))

def project(p, center):
    x, y, z = (p[i] - center[i] for i in range(3))
    return (W * .5 + (x + y) * .70710678 * scale,
            H * .60 + ((x - y) * .40824829 - z * .81649658) * scale)

def cluster_colors(voxels, bonds):
    parent = {voxel[0]: voxel[0] for voxel in voxels}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in bonds:
        if i in parent and j in parent:
            parent[find(j)] = find(i)
    roots = sorted({find(i) for i in parent})
    return {i: roots.index(find(i)) % len(palette) for i in parent}

def readable_event(label):
    if label == "fatigue_fracture_three_clusters":
        return "Fatigue fracture → three clusters"
    if label == "controlled_impulse_toward_fragment_B":
        return "Controlled impulse toward fragment B"
    if label.startswith("fragment_collision_heat_"):
        return "Fragment contact · restitution heat"
    return label.replace("_", " ")

for number, (step, frame) in enumerate(sorted(frames.items()), 1):
    active_events = [label for at, label in events if at <= step]
    restitution_heat = sum(heat for at, heat in heat_events if at <= step)
    ledger = frame["ledger"]
    mass, internal, kinetic = map(float, ledger[4:7])
    clusters = int(ledger[3])
    momentum = tuple(map(float, ledger[8:11]))
    points = [voxel[1:4] for voxel in frame["voxels"]]
    center = tuple(sum(p[axis] for p in points) / max(len(points), 1) for axis in range(3))
    image = Image.new("RGB", (W, H), (7, 16, 28))
    draw = ImageDraw.Draw(image)
    draw.text((54, 34), "FATIGUE FRACTURE · FRAGMENT COLLISION · 3D", font=font(26), fill=white)
    draw.text((58, 80), f"t = {step*dt:.2f} s · состояние {number}/{len(frames)} · шаг {step}/{steps}", font=font(18), fill=muted)

    by_id = {voxel[0]: voxel for voxel in frame["voxels"]}
    color_by_id = cluster_colors(frame["voxels"], frame["bonds"])
    for i, j in frame["bonds"]:
        if i in by_id and j in by_id:
            a, b = project(by_id[i][1:4], center), project(by_id[j][1:4], center)
            draw.line((a, b), fill=(209, 224, 238), width=9)
    for voxel in frame["voxels"]:
        index, x, y, z, temperature, damage = voxel
        cluster = color_by_id[index]
        color = palette[cluster]
        pts = [project((x+dx*.01, y+dy*.01, z+dz*.01), center) for dx,dy,dz in cube]
        for face, shade in zip(faces, (.68, .84, 1.12)):
            face_color = tuple(round(channel*shade) for channel in color)
            draw.polygon([pts[i] for i in face], fill=face_color, outline=(238, 244, 250))
        cx, cy = project((x, y, z), center)
        draw.text((cx-9, cy+88), chr(ord("A")+index), font=font(19), fill=white)
        if damage > 0:
            draw.text((cx-32, cy-118), f"d={damage:.2f}", font=font(14), fill=(255, 199, 126))

    draw.rounded_rectangle((58, 145, 770, 490), 18, fill=(17,29,44), outline=(61,82,103), width=2)
    draw.text((88, 170), f"Кластеры: {clusters} · связи в кадре: {len(frame['bonds'])}", font=font(23), fill=white)
    draw.text((88, 220), f"Масса: {mass:.9e} кг", font=font(20), fill=muted)
    draw.text((88, 260), f"Внутренняя энергия: {internal:.9e} Дж", font=font(19), fill=muted)
    draw.text((88, 298), f"Кинетическая энергия: {kinetic:.9e} Дж", font=font(19), fill=muted)
    draw.text((88, 337), f"Импульс: [{momentum[0]:.3e}, {momentum[1]:.3e}, {momentum[2]:.3e}] кг·м/с", font=font(17), fill=muted)
    draw.text((88, 375), f"Restitution heat: {restitution_heat:.6e} J", font=font(18), fill=(255,174,88))
    event_text = readable_event(active_events[-1]) if active_events else "Ожидание циклического нагружения"
    draw.text((88, 414), event_text, font=font(17), fill=white)
    draw.text((88, 453), f"Средняя X / COM: {center[0]:+.4f} m · камера следует за группой", font=font(17), fill=muted)
    images.append(image)
    if number % 32 == 0:
        print(f"render {number}/{len(frames)}", flush=True)

gif = prefix.with_suffix(".gif")
paletted = [image.convert("P", palette=Image.Palette.ADAPTIVE, colors=192) for image in images]
paletted[0].save(gif, save_all=True, append_images=paletted[1:], duration=100, loop=0, optimize=False)
mp4 = prefix.with_suffix(".mp4")
command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", "10", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(mp4)]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
try:
    for image in images:
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed writing MP4")
except BaseException:
    proc.kill()
    raise
print(f"rendered {len(frames)} saved physical states, GIF={gif}, MP4={mp4}")
