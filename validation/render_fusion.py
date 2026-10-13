#!/usr/bin/env python3
"""Render the measured iron/copper REV fusion trajectory to GIF and MP4."""
import csv
import subprocess
import sys
from pathlib import Path

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/fusion.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix("")
dt, steps, frames, fusion_step = 0.0, 0, {}, None
for row in csv.reader(source.open(newline="")):
    if not row:
        continue
    if row[0] == "H":
        dt, steps = float(row[1]), int(row[2])
    elif row[0] == "S":
        frames[int(row[1])] = {"voxels": {}, "ledger": None}
    elif row[0] == "V":
        frames[int(row[1])]["voxels"][int(row[2])] = {"p": tuple(map(float, row[3:6]))}
    elif row[0] == "Q":
        frames[int(row[1])]["voxels"].setdefault(int(row[2]), {})["temp"] = float(row[6])
    elif row[0] == "C":
        voxel = frames[int(row[1])]["voxels"].setdefault(int(row[2]), {})
        voxel["mass"] = float(row[5])
        voxel.setdefault("composition", {})[int(row[3])] = float(row[4])
    elif row[0] == "L":
        frames[int(row[1])]["ledger"] = row
    elif row[0] == "W":
        frames[int(row[1])]["external_work"] = float(row[2])
        frames[int(row[1])]["energy_residual"] = float(row[3]) if len(row) > 3 else 0.0
    elif row[0] == "E" and row[2] == "fusion":
        fusion_step = int(row[1])

if len(frames) < 176 or (max(frames) - min(frames)) * dt < 17.6 - 1e-9:
    raise SystemExit("fusion capture needs 176+ actual states over at least 17.6 seconds")
if fusion_step is None or any(frame["ledger"] is None for frame in frames.values()):
    raise SystemExit("fusion event or a per-frame conservation ledger is missing")

prefix.parent.mkdir(parents=True, exist_ok=True)
W, H, output = 1120, 700, []
iron_id, copper_id = 2, 6
for number, (step, frame) in enumerate(sorted(frames.items()), 1):
    voxels = frame["voxels"]
    cx0 = sum(v["p"][0] for v in voxels.values()) / len(voxels)
    cy0 = sum(v["p"][1] for v in voxels.values()) / len(voxels)
    centers = {i: (560 + (v["p"][0] - cx0) * 4200, 330 - (v["p"][1] - cy0) * 4200) for i, v in voxels.items()}
    ledger = frame["ledger"]
    mass, internal, kinetic, fusion_heat = map(float, [ledger[4], ledger[5], ledger[6], ledger[11]])
    external_work = frame.get("external_work", 0.0)
    energy_residual = frame.get("energy_residual", 0.0)
    px, py, pz = map(float, ledger[8:11])
    component_mass = {}
    for voxel in voxels.values():
        for component, fraction in voxel.get("composition", {}).items():
            component_mass[component] = component_mass.get(component, 0.0) + voxel.get("mass", 0.0) * fraction
    fe = component_mass.get(iron_id, 0.0) / mass
    cu = component_mass.get(copper_id, 0.0) / mass
    fused = step >= fusion_step
    elems = [f'<rect width="{W}" height="{H}" fill="#07101c"/>']
    elems.append('<defs><linearGradient id="alloy" x1="0" x2="1"><stop offset="0%" stop-color="#d8783a"/><stop offset="100%" stop-color="#bd8050"/></linearGradient></defs>')
    elems.append('<text x="40" y="42" fill="#ebf2fa" font-size="24" font-family="sans-serif">IRON + COPPER · REV FUSION · ENGINE SCENE API</text>')
    elems.append(f'<text x="42" y="75" fill="#99aec6" font-size="16" font-family="sans-serif">t = {step*dt:.2f} s · шаг {step}/{steps} · состояние {number}/{len(frames)} · слияние на шаге {fusion_step}</text>')
    elems.append('<line x1="250" y1="330" x2="870" y2="330" stroke="#556779" stroke-width="2" stroke-dasharray="8 9"/>')
    for i, voxel in sorted(voxels.items()):
        x, y = centers[i]
        composition = voxel.get("composition", {})
        fill = "url(#alloy)" if len(composition) > 1 else ("#d8783a" if iron_id in composition else "#bd8050")
        label = "Fe+Cu" if len(composition) > 1 else ("Fe" if iron_id in composition else "Cu")
        elems.append(f'<rect x="{x-20:.1f}" y="{y-20:.1f}" width="40" height="40" rx="6" fill="{fill}" stroke="#e8eef7" stroke-width="2"/>')
        elems.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" fill="#07101c" font-size="12" font-family="sans-serif">{label}</text>')
        elems.append(f'<text x="{x:.1f}" y="{y+34:.1f}" text-anchor="middle" fill="#ebf2fa" font-size="11" font-family="sans-serif">{voxel.get("temp", 0.0):.1f} K</text>')
    state = "один смешанный REV" if fused else "два отдельных REV"
    elems.append('<rect x="55" y="510" width="1010" height="145" rx="16" fill="#111d2c" stroke="#3d5267" stroke-width="2"/>')
    elems.append(f'<text x="82" y="546" fill="#42d6c4" font-size="20" font-family="sans-serif">Состояние: {state} · Fe {fe:.5f} · Cu {cu:.5f} · масса {mass:.9f} kg</text>')
    elems.append(f'<text x="82" y="580" fill="#ebf2fa" font-size="16" font-family="sans-serif">U {internal:.9e} J · K {kinetic:.9e} J · работа {external_work:.9e} J · остаток {energy_residual:.3e} J</text>')
    elems.append(f'<text x="82" y="636" fill="#e6b96b" font-size="13" font-family="sans-serif">Fusion heat {fusion_heat:.3e} J · работа — сумма F·Δx (с gravity) за substep; остаток не закрыт</text>')
    elems.append(f'<text x="82" y="612" fill="#99aec6" font-size="16" font-family="sans-serif">Импульс ({px:.3e}, {py:.3e}, {pz:.3e}) kg·m/s · разорвано связей {ledger[3]}</text>')
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
