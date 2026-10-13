#!/usr/bin/env python3
"""Render the actual hot-contact bond-formation trajectory to GIF and MP4."""
import csv
import subprocess
import sys
from pathlib import Path

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/bond-formation.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix("")
dt, steps, frames = 0.0, 0, {}
formed_step, heater_input = None, 0.0
for row in csv.reader(source.open(newline="")):
    if not row:
        continue
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "S":
        frames[int(row[1])] = {"voxels": {}, "bond": False, "ledger": None}
    elif row[0] == "V":
        frames[int(row[1])]["voxels"][int(row[2])] = {
            "p": tuple(map(float, row[3:6])), "damage": float(row[6])
        }
    elif row[0] == "Q":
        frames[int(row[1])]["voxels"].setdefault(int(row[2]), {})["temp"] = float(row[6])
    elif row[0] == "B":
        frames[int(row[1])]["bond"] = True
    elif row[0] == "L":
        frames[int(row[1])]["ledger"] = row
    elif row[0] == "E" and row[2] == "bond":
        formed_step = int(row[1])
    elif row[0] == "R" and row[1] == "setup":
        heater_input = float(row[2])

if len(frames) < 176 or (max(frames) - min(frames)) * dt < 17.6 - 1e-9:
    raise SystemExit(f"capture needs 176+ actual states and 17.6 s; got {len(frames)} and {(max(frames)-min(frames))*dt:.2f}s")
if formed_step is None or any(frame["ledger"] is None for frame in frames.values()):
    raise SystemExit("capture lacks a bond event or per-frame energy/momentum ledger")

prefix.parent.mkdir(parents=True, exist_ok=True)
W, H, output = 1120, 700, []
def esc(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
for number, (step, frame) in enumerate(sorted(frames.items()), 1):
    voxels = frame["voxels"]
    xs = [v["p"][0] for v in voxels.values()]
    mid = sum(xs) / len(xs)
    cx = {i: 560 + (v["p"][0] - mid) * 10000 for i, v in voxels.items()}
    ledger = frame["ledger"]
    internal, kinetic, pending = map(float, ledger[5:8])
    momentum = tuple(map(float, ledger[8:11]))
    distance = abs(xs[1] - xs[0])
    elems = [f'<rect width="{W}" height="{H}" fill="#07101c"/>']
    elems.append('<text x="40" y="42" fill="#ebf2fa" font-size="24" font-family="sans-serif">HOT CONTACT · BOND FORMATION · ENGINE SOLVER</text>')
    elems.append(f'<text x="42" y="75" fill="#99aec6" font-size="16" font-family="sans-serif">t = {step*dt:.2f} s · шаг {step}/{steps} · состояние {number}/{len(frames)}</text>')
    elems.append('<line x1="250" y1="430" x2="870" y2="430" stroke="#556779" stroke-width="3"/>')
    if frame["bond"] and len(cx) == 2:
        elems.append(f'<line x1="{cx[0]:.1f}" y1="330" x2="{cx[1]:.1f}" y2="330" stroke="#42d6c4" stroke-width="12"/>')
        elems.append(f'<text x="{(cx[0]+cx[1])/2-42:.1f}" y="300" fill="#42d6c4" font-size="16" font-family="sans-serif">bond</text>')
    for i, voxel in sorted(voxels.items()):
        x = cx[i]
        t = max(0.0, min(1.0, (voxel.get("temp", 293.15)-293.15)/500.0))
        color = f'#{int(55+195*t):02x}{int(135-60*t):02x}{int(225-160*t):02x}'
        top = f'{x:.1f},290 {x+48:.1f},315 {x:.1f},340 {x-48:.1f},315'
        left = f'{x-48:.1f},315 {x:.1f},340 {x:.1f},430 {x-48:.1f},405'
        right = f'{x:.1f},340 {x+48:.1f},315 {x+48:.1f},405 {x:.1f},430'
        elems.append(f'<polygon points="{left}" fill="#273f60" stroke="#d7e2ee"/>')
        elems.append(f'<polygon points="{right}" fill="#36557d" stroke="#d7e2ee"/>')
        elems.append(f'<polygon points="{top}" fill="{color}" stroke="#f0f5fa"/>')
        elems.append(f'<text x="{x-20:.1f}" y="455" fill="#ebf2fa" font-size="16" font-family="sans-serif">REV {i}</text>')
        elems.append(f'<text x="{x-33:.1f}" y="270" fill="#ebf2fa" font-size="15" font-family="sans-serif">{voxel.get("temp",293.15):.1f} K</text>')
    state = f"образована на шаге {formed_step}" if frame["bond"] and step >= formed_step else "раздельные REVs · сближение" if step < formed_step else "контакт"
    elems.append('<rect x="55" y="510" width="1010" height="145" rx="16" fill="#111d2c" stroke="#3d5267" stroke-width="2"/>')
    elems.append(f'<text x="82" y="546" fill="#42d6c4" font-size="20" font-family="sans-serif">Связь: {esc(state)} · расстояние центров {distance*1000:.4f} mm</text>')
    elems.append(f'<text x="82" y="580" fill="#ebf2fa" font-size="16" font-family="sans-serif">Внутренняя энергия {internal:.8e} J · кинетическая {kinetic:.8e} J · pending fracture {pending:.3e} J</text>')
    elems.append(f'<text x="82" y="612" fill="#99aec6" font-size="16" font-family="sans-serif">Нагреватель {heater_input:.6f} J · импульс ({momentum[0]:.3e}, {momentum[1]:.3e}, {momentum[2]:.3e}) kg·m/s</text>')
    svg = prefix.parent / f"{prefix.name}-{number:03}.svg"
    svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">' + ''.join(elems) + '</svg>', encoding="utf-8")
    output.append(svg)

sequence = str(prefix.parent / f"{prefix.name}-%03d.svg")
mp4, gif = prefix.with_suffix(".mp4"), prefix.with_suffix(".gif")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", sequence, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4)], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", sequence, "-filter_complex", "[0:v]split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer", "-loop", "0", str(gif)], check=True)
for path in output:
    path.unlink()
print(f"rendered {len(output)} actual states, physical={(max(frames)-min(frames))*dt:.2f}s, GIF={gif}, MP4={mp4}")
