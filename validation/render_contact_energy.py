#!/usr/bin/env python3
"""Render recorded free-body collision and restitution heat in isometric 3D."""
import csv, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

source=Path(sys.argv[1]) if len(sys.argv)>1 else Path("validation/contact-energy.csv")
prefix=Path(sys.argv[2]) if len(sys.argv)>2 else source.with_suffix("")
dt,steps=0.01,0; frames={}; impacts=[]
for row in csv.reader(source.open(newline="")):
    if not row: continue
    if row[0]=="H": dt,steps=float(row[1]),int(row[2])
    elif row[0]=="S": frames[int(row[1])] = {"voxels":[],"ledger":None}
    elif row[0]=="V": frames[int(row[1])]["voxels"].append(tuple(map(float,row[2:])))
    elif row[0]=="L": frames[int(row[1])]["ledger"]=row
    elif row[0]=="E": impacts.append((int(row[1]),row[3]))
if len(frames)<176 or max(frames)*dt<17.6-1e-9 or any(f["ledger"] is None for f in frames.values()):
    raise SystemExit("capture needs 176+ actual states, 17.6 s physical time, and per-frame ledger")
font=lambda size:ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",size)
W,H=1600,1000; white=(235,242,250); muted=(153,174,198); output=[]
scale=1450.0
def project(x,y,z,mid): return (800+(x-mid)*scale-y*scale*.35,620-y*scale*.2-z*scale)
impact_step=impacts[0][0] if impacts else None
heat_value=float(impacts[0][1].split("_")[2]) if impacts else 0.0
for frame_no,(step,f) in enumerate(sorted(frames.items()),1):
    ledger=f["ledger"]; mid=sum(v[1] for v in f["voxels"])/max(len(f["voxels"]),1)
    im=Image.new("RGB",(W,H),(7,16,28));d=ImageDraw.Draw(im)
    d.text((55,35),"FREE REV COLLISION · RESTITUTION · ENERGY LEDGER",font=font(25),fill=white)
    d.text((58,78),f"t = {step*dt:.2f} s · кадр {frame_no}/{len(frames)} · физический шаг {step}/{steps}",font=font(17),fill=muted)
    for v in f["voxels"]:
        i,x,y,z,_,heat= int(v[0]),*v[1:]
        cx,cy=project(x,y,z,mid); level=min(1.0,heat/max(heat_value/2,1e-20))
        color=(round(68+185*level),round(139+42*level),round(208-130*level))
        top=[(cx,cy-48),(cx+52,cy-20),(cx,cy+8),(cx-52,cy-20)]
        left=[top[3],top[2],(cx,cy+64),(cx-52,cy+36)]
        right=[top[2],top[1],(cx+52,cy+36),(cx,cy+64)]
        d.polygon(left,fill=tuple(int(c*.64) for c in color),outline=(215,226,238))
        d.polygon(right,fill=tuple(int(c*.82) for c in color),outline=(215,226,238))
        d.polygon(top,fill=color,outline=(240,245,250));d.text((cx-8,cy+70),str(i),font=font(16),fill=white)
    internal_j=float(ledger[5]); kinetic_j=float(ledger[6]); mass=float(ledger[4])
    d.rounded_rectangle((95,150,825,356),18,fill=(17,29,44),outline=(61,82,103),width=2)
    d.text((125,175),f"Restitution heat: {heat_value:.3e} J",font=font(24),fill=(255,174,88))
    d.text((125,222),f"Contact: {'impact at step '+str(impact_step) if impact_step and step>=impact_step else 'approaching'}",font=font(20),fill=white)
    d.text((125,266),f"Internal: {internal_j:.8e} J    Kinetic: {kinetic_j:.8e} J",font=font(17),fill=muted)
    d.text((125,303),f"Mass: {mass:.7f} kg · initial temperature: 293.15 K · friction heat: 0 J",font=font(17),fill=muted)
    path=prefix.parent/f"{prefix.name}-{frame_no:03}.png"; im.save(path);output.append(path)
gif=prefix.with_suffix(".gif")
images=[Image.open(p).convert("P",palette=Image.Palette.ADAPTIVE,colors=192) for p in output]
images[0].save(gif,save_all=True,append_images=images[1:],duration=100,loop=0,optimize=False)
mp4=prefix.with_suffix(".mp4")
subprocess.run(["ffmpeg","-y","-v","error","-framerate","10","-i",str(prefix.parent/f"{prefix.name}-%03d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",str(mp4)],check=True)
print(f"rendered {len(output)} actual states, media={len(output)/10:.1f}s, physical={(max(frames)-min(frames))*dt:.1f}s")
