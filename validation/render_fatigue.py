#!/usr/bin/env python3
"""Render recorded constitutive fatigue states to a 3D GIF and MP4."""
import csv
import math
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/fatigue-cycles.csv")
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix("")
dt, steps, frames = 0.01, 0, {}
for row in csv.reader(source.open(newline="")):
    if not row: continue
    if row[0] == "H": dt, steps = float(row[1]), int(row[2])
    elif row[0] == "S": frames[int(row[1])] = {"voxels": [], "bonds": []}
    elif row[0] == "V":
        frames[int(row[1])]["voxels"].append(tuple(map(float, row[2:])))
    elif row[0] == "B": frames[int(row[1])]["bonds"].append((int(row[2]), int(row[3])))

if len(frames) < 176 or max(frames) * dt < 17.6 - 1e-9:
    raise SystemExit(f"incomplete physical capture: {len(frames)} frames, {max(frames) * dt:.2f}s")
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = lambda size: ImageFont.truetype(font_path, size)
W, H = 1600, 1000
white, muted = (235, 242, 250), (153, 174, 198)
output = []
prefix.parent.mkdir(parents=True, exist_ok=True)

def project(x, y, z):
    return 800 + (x-y)*0.7071*22000, 610 + ((x+y)*0.4082-z*0.8165)*22000

for number, (step, frame) in enumerate(sorted(frames.items()), 1):
    image = Image.new("RGB", (W, H), (7, 16, 28)); d = ImageDraw.Draw(image)
    d.text((55, 35), "CYCLIC FATIGUE · IRON REV BOND · ENGINE CONSTITUTIVE UPDATE", font=font(25), fill=white)
    d.text((58, 78), f"t = {step*dt:.2f} s · кадр {number}/{len(frames)} · физический шаг {step}/{steps}", font=font(17), fill=muted)
    d.line((330, 800, 1270, 800), fill=(67, 89, 108), width=2)
    by_id = {int(v[0]): v for v in frame["voxels"]}
    for a, b in frame["bonds"]:
        pa, pb = by_id[a], by_id[b]
        fatigue = max(pa[5], pb[5])
        color = (round(75+175*fatigue), round(175-115*fatigue), round(232-180*fatigue))
        d.line((project(*pa[1:4]), project(*pb[1:4])), fill=color, width=16)
    for v in sorted(frame["voxels"], key=lambda v:v[3]):
        idx, x, y, z, damage, fatigue = int(v[0]), *v[1:]
        heat = max(damage, fatigue)
        color = (round(75+175*heat), round(165-100*heat), round(220-160*heat))
        cx, cy = project(x,y,z); h=50
        top=[(cx,cy-h),(cx+55,cy-h+30),(cx,cy-h+60),(cx-55,cy-h+30)]
        left=[top[3],top[2],(cx,cy+50),(cx-55,cy+20)]
        right=[top[2],top[1],(cx+55,cy+20),(cx,cy+50)]
        d.polygon(left,fill=tuple(int(c*.65) for c in color),outline=(215,226,238))
        d.polygon(right,fill=tuple(int(c*.82) for c in color),outline=(215,226,238))
        d.polygon(top,fill=color,outline=(240,245,250))
        d.text((cx-10,cy+60),str(idx),font=font(16),fill=white)
    damage=max((v[5] for v in frame["voxels"]),default=0.0)
    d.rounded_rectangle((102,160,650,318),18,fill=(17,29,44),outline=(61,82,103),width=2)
    d.text((130,185),f"Усталостное повреждение: {damage:.3f}",font=font(24),fill=(255,190,90))
    d.text((130,232),f"Циклы: {step/10:.0f} · связь: {'целая' if frame['bonds'] else 'разорвана'}",font=font(20),fill=white)
    d.text((130,274),"Синусоидальная деформация · амплитуда ниже статической прочности",font=font(16),fill=muted)
    path=prefix.parent/f"{prefix.name}-{number:03}.png"; image.save(path); output.append(path)

gif=prefix.with_suffix(".gif")
images=[Image.open(p).convert("P",palette=Image.Palette.ADAPTIVE,colors=192) for p in output]
images[0].save(gif,save_all=True,append_images=images[1:],duration=100,loop=0,optimize=False)
mp4=prefix.with_suffix(".mp4")
subprocess.run(["ffmpeg","-y","-v","error","-framerate","10","-i",str(prefix.parent/f"{prefix.name}-%03d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",str(mp4)],check=True)
print(f"rendered {len(output)} real states; GIF={gif}; MP4={mp4}; media={len(output)/10:.1f}s; physical={(max(frames)-min(frames))*dt:.2f}s")
