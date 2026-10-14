#!/usr/bin/env python3
"""Render actual saved states from the liquid shear viscosity capture."""
import csv
import math
import subprocess
import sys
from pathlib import Path

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/liquid-viscosity.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix("")
dt, steps, frames = 0.0, 0, {}
mu_reference, reference_temperature, activation_energy = 1.002e-3, 293.15, 0.0
interaction = "liquid-liquid"
for row in csv.reader(source.open(newline="")):
    if not row:
        continue
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "R" and row[1] == "setup" and len(row) >= 7:
        mu_reference, _, _, reference_temperature, activation_energy = map(float, row[2:7])
    elif row[0] == "R" and row[1] == "interaction" and len(row) >= 3:
        interaction = row[2]
    elif row[0] == "S":
        frames[int(row[1])] = {"voxels": {}, "ledger": None}
    elif row[0] == "Q":
        frames[int(row[1])]["voxels"][int(row[2])] = {
            "p": tuple(map(float, row[3:6])), "temp": float(row[6]), "liquid": float(row[7])
        }
    elif row[0] == "L":
        frames[int(row[1])]["ledger"] = row

if len(frames) < 176 or (max(frames) - min(frames)) * dt < 17.6:
    raise SystemExit("capture must contain 176+ actual states over at least 17.6 seconds")
if any(frame["ledger"] is None for frame in frames.values()):
    raise SystemExit("capture lacks per-frame conservation data")

prefix.parent.mkdir(parents=True, exist_ok=True)
W, H, output = 1120, 700, []
for number, (step, frame) in enumerate(sorted(frames.items()), 1):
    voxels = frame["voxels"]
    # World-space shear displacement is intentionally retained and scaled up
    # for visibility. Each frame is still drawn from its actual CSV positions.
    center_x = sum(v["p"][0] for v in voxels.values()) / len(voxels)
    center_y = sum(v["p"][1] for v in voxels.values()) / len(voxels)
    cx = {i: 560 + (v["p"][0] - center_x) * 1800 for i, v in voxels.items()}
    cy = {i: 330 - (v["p"][1] - center_y) * 1800 for i, v in voxels.items()}
    ledger = frame["ledger"]
    internal, kinetic, heat = map(float, ledger[5:8])
    momentum = tuple(map(float, ledger[8:11]))
    speed = abs(voxels[1]["p"][1] - voxels[0]["p"][1]) / max(step * dt, dt)
    temperature = sum(v["temp"] for v in voxels.values()) / len(voxels)
    exponent = (activation_energy / 8.31446261815324) * (1.0 / temperature - 1.0 / reference_temperature)
    viscosity = mu_reference * math.exp(max(-80.0, min(80.0, exponent)))
    elems = [f'<rect width="{W}" height="{H}" fill="#07101c"/>']
    title = "LIQUID WATER · SOLID CONTACT · VISCOUS SHEAR" if interaction == "liquid-solid" else "LIQUID WATER · LIQUID CONTACT · VISCOUS SHEAR"
    elems.append(f'<text x="40" y="42" fill="#ebf2fa" font-size="24" font-family="sans-serif">{title}</text>')
    elems.append(f'<text x="42" y="75" fill="#99aec6" font-size="16" font-family="sans-serif">t = {step*dt:.2f} s · шаг {step}/{steps} · состояние {number}/{len(frames)} · T = {temperature:.2f} K · μ(T) = {viscosity*1000:.3f} mPa·s</text>')
    elems.append('<line x1="170" y1="330" x2="950" y2="330" stroke="#556779" stroke-width="2" stroke-dasharray="8 9"/>')
    for i, voxel in sorted(voxels.items()):
        x, y = cx[i], cy[i]
        color = "#33b8d1" if voxel["liquid"] > 0.5 else "#d9803b"
        elems.append(f'<rect x="{x-68:.1f}" y="{y-68:.1f}" width="136" height="136" rx="18" fill="{color}" stroke="#e8eef7" stroke-width="3"/>')
        elems.append(f'<text x="{x-35:.1f}" y="{y+7:.1f}" fill="#07101c" font-size="20" font-family="sans-serif">REV {i}</text>')
        elems.append(f'<text x="{x-48:.1f}" y="{y+100:.1f}" fill="#ebf2fa" font-size="15" font-family="sans-serif">{voxel["temp"]:.2f} K · жидкость {voxel["liquid"]:.2f}</text>')
    elems.append('<rect x="55" y="510" width="1010" height="145" rx="16" fill="#111d2c" stroke="#3d5267" stroke-width="2"/>')
    elems.append(f'<text x="82" y="546" fill="#42d6c4" font-size="20" font-family="sans-serif">Касательное течение · скорость сдвига {speed:.6e} m/s · вязкое тепло {heat:.6e} J</text>')
    elems.append(f'<text x="82" y="580" fill="#ebf2fa" font-size="16" font-family="sans-serif">Внутренняя энергия {internal:.12e} J · кинетическая {kinetic:.12e} J · масса {float(ledger[4]):.9f} kg</text>')
    elems.append(f'<text x="82" y="612" fill="#99aec6" font-size="16" font-family="sans-serif">Импульс ({momentum[0]:.3e}, {momentum[1]:.3e}, {momentum[2]:.3e}) kg·m/s · разорвано связей {ledger[3]}</text>')
    svg = prefix.parent / f"{prefix.name}-{number:03}.svg"
    svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">' + "".join(elems) + "</svg>", encoding="utf-8")
    output.append(svg)

sequence = str(prefix.parent / f"{prefix.name}-%03d.svg")
mp4, gif = prefix.with_suffix(".mp4"), prefix.with_suffix(".gif")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", sequence, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4)], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", sequence, "-filter_complex", "[0:v]split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer", "-loop", "0", str(gif)], check=True)
for path in output:
    path.unlink()
print(f"rendered {len(output)} actual states, physical={(max(frames)-min(frames))*dt:.2f}s, GIF={gif}, MP4={mp4}")
