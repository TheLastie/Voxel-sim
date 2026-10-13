#!/usr/bin/env python3
"""Render actual thermal-strength states to SVG-backed GIF and MP4."""
import csv
import subprocess
import sys
from pathlib import Path

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/thermal-strength.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix("")
dt, steps, frames = 0.0, 0, {}
material_name = "MATERIAL"
for row in csv.reader(source.open(newline="")):
    if not row:
        continue
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "N":
        material_name = row[1].split("_")[0].upper()
    elif row[0] == "S":
        frames[int(row[1])] = {"voxels": {}, "bond": False, "metrics": None}
    elif row[0] == "V":
        voxel = frames[int(row[1])]["voxels"].setdefault(int(row[2]), {})
        voxel["p"] = tuple(map(float, row[3:6]))
        voxel["damage"] = float(row[6])
    elif row[0] == "Q":
        frames[int(row[1])]["voxels"].setdefault(int(row[2]), {})["temp"] = float(row[6])
    elif row[0] == "B":
        frames[int(row[1])]["bond"] = True
    elif row[0] == "R":
        frames[int(row[1])]["metrics"] = tuple(map(float, row[2:6]))

if len(frames) < 176 or (max(frames) - min(frames)) * dt < 17.6 - 1e-9:
    raise SystemExit(f"incomplete physical capture: {len(frames)} states, {(max(frames)-min(frames))*dt:.2f}s")
prefix.parent.mkdir(parents=True, exist_ok=True)
width, height = 1600, 1000
outputs = []

def project(point):
    x, y, z = point
    return 800 + (x-y)*0.7071*20_000, 610 + ((x+y)*0.4082-z*0.8165)*20_000

for number, (step, frame) in enumerate(sorted(frames.items()), 1):
    temp, factor, stress, heat = frame["metrics"] or (300.0, 1.0, 0.0, 0.0)
    elements = [f'<rect width="{width}" height="{height}" fill="#07101c"/>']
    elements.append(f'<text x="55" y="52" fill="#ebf2fa" font-size="25" font-family="sans-serif">THERMAL SOFTENING · {material_name} BOND · ENGINE SOLVER</text>')
    elements.append(f'<text x="58" y="92" fill="#99aec6" font-size="18" font-family="sans-serif">t = {step*dt:.2f} s · physical step {step}/{steps} · state {number}/{len(frames)}</text>')
    points = {idx: project(voxel["p"]) for idx, voxel in frame["voxels"].items()}
    if frame["bond"] and 0 in points and 1 in points:
        a, b = points[0], points[1]
        elements.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#64d2f0" stroke-width="18"/>')
    for idx, voxel in sorted(frame["voxels"].items()):
        cx, cy = points[idx]
        t = max(0.0, min(1.0, (voxel.get("temp", 300.0)-300.0)/1500.0))
        color = f'#{int(50+205*t):02x}{int(130-80*t):02x}{int(235-190*t):02x}'
        top = f'{cx:.1f},{cy-80:.1f} {cx+90:.1f},{cy-30:.1f} {cx:.1f},{cy+20:.1f} {cx-90:.1f},{cy-30:.1f}'
        left = f'{cx-90:.1f},{cy-30:.1f} {cx:.1f},{cy+20:.1f} {cx:.1f},{cy+130:.1f} {cx-90:.1f},{cy+80:.1f}'
        right = f'{cx:.1f},{cy+20:.1f} {cx+90:.1f},{cy-30:.1f} {cx+90:.1f},{cy+80:.1f} {cx:.1f},{cy+130:.1f}'
        elements.append(f'<polygon points="{left}" fill="#253b5b" stroke="#d7e2ee"/>')
        elements.append(f'<polygon points="{right}" fill="#325078" stroke="#d7e2ee"/>')
        elements.append(f'<polygon points="{top}" fill="{color}" stroke="#f0f5fa"/>')
        elements.append(f'<text x="{cx-8:.1f}" y="{cy+160:.1f}" fill="#ebf2fa" font-size="20" font-family="sans-serif">REV {idx}</text>')
        elements.append(f'<text x="{cx-60:.1f}" y="{cy-100:.1f}" fill="#ebf2fa" font-size="18" font-family="sans-serif">{voxel.get("temp",300):.0f} K</text>')
    elements.append('<rect x="95" y="160" width="605" height="235" rx="18" fill="#111d2c" stroke="#3d5267" stroke-width="2"/>')
    elements.append(f'<text x="125" y="205" fill="#ffbe5a" font-size="27" font-family="sans-serif">Прочность связей: {factor:.3f}×</text>')
    elements.append(f'<text x="125" y="258" fill="#ebf2fa" font-size="21" font-family="sans-serif">Нагрузка / текущий предел: {stress:.3f}</text>')
    state = "сохранена" if frame["bond"] else "разрушена"
    elements.append(f'<text x="125" y="302" fill="#ebf2fa" font-size="21" font-family="sans-serif">Связь: {state}</text>')
    elements.append(f'<text x="125" y="346" fill="#99aec6" font-size="19" font-family="sans-serif">Подведённое тепло: {heat:.2f} J</text>')
    elements.append('<text x="95" y="925" fill="#99aec6" font-size="17" font-family="sans-serif">70% Tпл onset · линейное размягчение до Tпл · 0.8× начального предела</text>')
    path = prefix.parent / f"{prefix.name}-{number:03}.svg"
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + ''.join(elements) + '</svg>', encoding="utf-8")
    outputs.append(path)

mp4 = prefix.with_suffix(".mp4")
gif = prefix.with_suffix(".gif")
sequence = str(prefix.parent / f"{prefix.name}-%03d.svg")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", sequence, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4)], check=True)
subprocess.run(
    [
        "ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", sequence,
        "-filter_complex", "[0:v]split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer",
        "-loop", "0", str(gif),
    ],
    check=True,
)
for path in outputs:
    path.unlink()
print(f"rendered {len(outputs)} real states; GIF={gif}; MP4={mp4}; media={len(outputs)/10:.1f}s; physical={(max(frames)-min(frames))*dt:.2f}s")
