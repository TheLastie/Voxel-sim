#!/usr/bin/env python3
"""Render actual quasi-static dielectric sweep states to GIF and MP4."""
import csv
import subprocess
import sys
from pathlib import Path

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/dielectric-capacitance.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix("")
dt, steps, frames = 0.0, 0, {}
for row in csv.reader(source.open(newline="")):
    if not row:
        continue
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "S":
        frames[int(row[1])] = {"voxels": {}, "bonds": [], "metrics": None}
    elif row[0] == "V":
        frames[int(row[1])]["voxels"].setdefault(int(row[2]), {})["p"] = tuple(map(float, row[3:6]))
    elif row[0] == "Q":
        frames[int(row[1])]["voxels"].setdefault(int(row[2]), {})["temp"] = float(row[6])
    elif row[0] == "B":
        frames[int(row[1])]["bonds"].append((int(row[2]), int(row[3])))
    elif row[0] == "R":
        frames[int(row[1])]["metrics"] = tuple(map(float, row[2:7]))

if len(frames) < 176 or (max(frames) - min(frames)) * dt < 17.6 - 1e-9:
    raise SystemExit(f"incomplete physical capture: {len(frames)} states, {(max(frames)-min(frames))*dt:.2f}s")
prefix.parent.mkdir(parents=True, exist_ok=True)
width, height, outputs = 1600, 1000, []

def project(point):
    x, y, z = point
    return 800 + (x-y)*0.7071*20_000, 625 + ((x+y)*0.4082-z*0.8165)*20_000

for number, (step, frame) in enumerate(sorted(frames.items()), 1):
    voltage, charge, capacitance, stored_energy, source_work = frame["metrics"] or (0, 0, 0, 0, 0)
    elements = [f'<rect width="{width}" height="{height}" fill="#07101c"/>']
    elements.append('<text x="55" y="52" fill="#ebf2fa" font-size="25" font-family="sans-serif">DIELECTRIC RESPONSE · GLASS REV CAPACITOR · QUASI-STATIC SWEEP</text>')
    elements.append(f'<text x="58" y="92" fill="#99aec6" font-size="18" font-family="sans-serif">t = {step*dt:.2f} s · physical step {step}/{steps} · state {number}/{len(frames)}</text>')
    points = {idx: project(voxel["p"]) for idx, voxel in frame["voxels"].items()}
    for a, b in frame["bonds"]:
        p, q = points[a], points[b]
        elements.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="#c792ea" stroke-width="14"/>')
    for idx, voxel in sorted(frame["voxels"].items()):
        cx, cy = points[idx]
        t = max(0.0, min(1.0, voltage))
        color = f'#{int(70+175*t):02x}{int(105-25*t):02x}{int(215-25*t):02x}'
        top = f'{cx:.1f},{cy-75:.1f} {cx+82:.1f},{cy-30:.1f} {cx:.1f},{cy+15:.1f} {cx-82:.1f},{cy-30:.1f}'
        left = f'{cx-82:.1f},{cy-30:.1f} {cx:.1f},{cy+15:.1f} {cx:.1f},{cy+115:.1f} {cx-82:.1f},{cy+70:.1f}'
        right = f'{cx:.1f},{cy+15:.1f} {cx+82:.1f},{cy-30:.1f} {cx+82:.1f},{cy+70:.1f} {cx:.1f},{cy+115:.1f}'
        elements.append(f'<polygon points="{left}" fill="#253b5b" stroke="#d7e2ee"/>')
        elements.append(f'<polygon points="{right}" fill="#325078" stroke="#d7e2ee"/>')
        elements.append(f'<polygon points="{top}" fill="{color}" stroke="#f0f5fa"/>')
        elements.append(f'<text x="{cx-40:.1f}" y="{cy+145:.1f}" fill="#ebf2fa" font-size="18" font-family="sans-serif">REV {idx}</text>')
        elements.append(f'<text x="{cx-50:.1f}" y="{cy-95:.1f}" fill="#ebf2fa" font-size="17" font-family="sans-serif">{voxel.get("temp",293.15):.2f} K</text>')
    elements.append('<rect x="95" y="160" width="760" height="300" rx="18" fill="#111d2c" stroke="#3d5267" stroke-width="2"/>')
    elements.append(f'<text x="125" y="205" fill="#d5a6ff" font-size="27" font-family="sans-serif">ΔV = {voltage:.3f} V · Q = {charge*1e12:.4f} pC</text>')
    elements.append(f'<text x="125" y="258" fill="#ebf2fa" font-size="21" font-family="sans-serif">C = Q/ΔV = {capacitance*1e12:.6f} pF</text>')
    elements.append(f'<text x="125" y="302" fill="#ebf2fa" font-size="21" font-family="sans-serif">Полевая энергия = {stored_energy*1e15:.4f} fJ</text>')
    elements.append(f'<text x="125" y="346" fill="#99aec6" font-size="19" font-family="sans-serif">Работа источника = {source_work*1e15:.4f} fJ</text>')
    elements.append('<text x="95" y="925" fill="#99aec6" font-size="17" font-family="sans-serif">εᵣ стекла = 7 · три REV в серии · заряд и энергия поля выводятся из решённого потенциала</text>')
    path = prefix.parent / f"{prefix.name}-{number:03}.svg"
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + ''.join(elements) + '</svg>', encoding="utf-8")
    outputs.append(path)

sequence = str(prefix.parent / f"{prefix.name}-%03d.svg")
mp4, gif = prefix.with_suffix(".mp4"), prefix.with_suffix(".gif")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", sequence, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4)], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", sequence, "-filter_complex", "[0:v]split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer", "-loop", "0", str(gif)], check=True)
for path in outputs:
    path.unlink()
print(f"rendered {len(outputs)} real states; GIF={gif}; MP4={mp4}; media={len(outputs)/10:.1f}s; physical={(max(frames)-min(frames))*dt:.2f}s")
