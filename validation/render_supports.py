#!/usr/bin/env python3
"""Render real support_capture states; no scripted motion or interpolation."""
import csv, math, json, sys, subprocess
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
src=Path(sys.argv[1]); out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
frames={};floor=[];cells=[]
for r in csv.reader(src.open()):
 if r[0]=='H':dt=float(r[1]);steps=int(r[2])
 elif r[0]=='M':mount=list(map(float,r[1:]));
 elif r[0]=='T':cells.append(tuple(map(float,r[1:])))
 elif r[0]=='P':floor.append(tuple(map(float,r[1:])))
 elif r[0]=='F':frames[int(r[1])]={'f':list(map(float,r[2:])),'v':[]}
 elif r[0]=='V':frames[int(r[1])]['v'].append(list(map(float,r[2:])))
assert len(floor)==6400
assert all(len(f['v'])==4347 for f in frames.values())
assert max(f['f'][11] for f in frames.values())<1e-6
assert all(all(math.isfinite(a) for a in v) for f in frames.values() for v in f['v'])
W,H=1600,1000
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=lambda n:ImageFont.truetype(fontpath,n)
white=(234,241,249);muted=(149,168,190);blue=(80,181,248);orange=(243,168,89)
scale=950
origin_x,origin_y=570,650
def iso(p):
 x,y,z=p
 return((x+y)*.70710678,(x-y)*.40824829-z*.81649658)
# Fixed camera covers the entire floor, tree, guide and falling coupon.
def proj(p):
 x,y,z=p
 u,v=iso(p)
 return(origin_x+u*scale,origin_y+v*scale)
def rot(q,p):
 x,y,z,w=q; vx,vy,vz=p
 tx=2*(y*vz-z*vy);ty=2*(z*vx-x*vz);tz=2*(x*vy-y*vx)
 return(vx+w*tx+y*tz-z*ty,vy+w*ty+z*tx-x*tz,vz+w*tz+x*ty-y*tx)
def cube(d,p,color,q=None):
 candidates=[((1,0,0),[(1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)],.72),((0,-1,0),[(-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1)],.87),((0,0,1),[(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)],1.12)]
 for normal,face,factor in candidates:
  vertices=[]
  for v in face:
   v=tuple(a*.0049 for a in v);v=rot(q,v) if q else v
   vertices.append(proj(tuple(p[a]+v[a] for a in range(3))))
  d.polygon(vertices,fill=tuple(min(255,int(a*factor)) for a in color),outline=tuple(int(a*.5) for a in color))
ordered=sorted(frames); stats=[]
for index,n in enumerate(ordered):
 f=frames[n];pos=f['f'][:3];q=f['f'][3:7];t=n*dt
 # Fixed orientation; fit camera to real world extents, never alter body poses.
 view=[(-.4,-.4,0),(.4,-.4,0),(.4,.4,0),(-.4,.4,0),(0,0,.5),(-.43,0,.15),(.35,0,.15)]
 view.extend(tuple(v[1:4]) for v in f['v'] if abs(v[1])<.65 and abs(v[2])<.55 and -.04<v[3]<.6)
 view.extend(tuple(pos[a]+rot(q,c)[a] for a in range(3)) for c in cells)
 uv=[iso(p) for p in view]
 lo=[min(p[a] for p in uv) for a in range(2)];hi=[max(p[a] for p in uv) for a in range(2)]
 scale=min(1000/(hi[0]-lo[0]+.06),615/(hi[1]-lo[1]+.06),950)
 origin_x=565-(lo[0]+hi[0])/2*scale
 origin_y=557-(lo[1]+hi[1])/2*scale
 im=Image.new('RGB',(W,H),(11,18,29));d=ImageDraw.Draw(im)
 d.text((45,24),'ПОЛ · ГРАВИТАЦИЯ · ЗАКРЕПЛЁННЫЙ ТОПОР',font=font(29),fill=white)
 d.text((47,68),'Состояния реального решателя • 1 воксель = 1 см • воспроизведение замедлено в 5 раз',font=font(18),fill=muted)
 d.rounded_rectangle((30,111,1100,949),20,fill=(16,26,40),outline=(43,61,79),width=2)
 # Uniform fixed floor is drawn as its actual top faces, aggregated only for rendering.
 bounds=[min(p[a] for p in floor)-.005 for a in range(2)]+[max(p[a] for p in floor)+.005 for a in range(2)]
 x0,y0,x1,y1=bounds
 d.polygon([proj(p) for p in [(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0)]],fill=(43,59,74),outline=(101,125,145))
 for j in range(81):
  x=x0+j*.01;y=y0+j*.01
  d.line([proj((x,y0,0)),proj((x,y1,0))],fill=(51,69,84),width=1)
  d.line([proj((x0,y,0)),proj((x1,y,0))],fill=(51,69,84),width=1)
 # Static rail, world boundary fixture. Blue base moves only along this rail.
 base=tuple(pos[a]+rot(q,mount[3:])[a] for a in range(3))
 rail_y=mount[1]+.045
 d.line([proj((-.43,rail_y,mount[2])),proj((max(.35,base[0]+.08),rail_y,mount[2]))],fill=blue,width=6)
 for x in [-.35,.3]:
  d.line([proj((x,rail_y,0)),proj((x,rail_y,mount[2]))],fill=(44,104,148),width=5)
 items=[]
 for v in f['v']:
  i=int(v[0]);p=tuple(v[1:4]);damage=v[4];lost=v[5]
  color=blue if i>=4320 else ((217,74,58) if lost>=2 else (240,142,68) if lost else (int(147+92*damage),int(86+87*damage),int(38+24*damage)))
  items.append((p[0]-p[1]+p[2],p,color,None))
 for c in cells:
  off=rot(q,c);p=tuple(pos[a]+off[a] for a in range(3))
  items.append((p[0]-p[1]+p[2],p,(187,197,208),q))
 for _,p,color,r in sorted(items,key=lambda i:i[0]):
  sx,sy=proj(p)
  if 55<sx<1075 and 242<sy<868:cube(d,p,color,r)
 # Every REV remains represented in the overview, including bodies leaving the finite floor.
 overview=[p for _,p,_,_ in items]+[(-.4,-.4,0),(.4,.4,0),(-.4,.4,0),(.4,-.4,0)]
 ov=[iso(p) for p in overview];ol=[min(p[a] for p in ov) for a in range(2)];oh=[max(p[a] for p in ov) for a in range(2)]
 os=min(260/(oh[0]-ol[0]+.02),100/(oh[1]-ol[1]+.02))
 def op(p):
  u,v=iso(p);return(931+(u-(ol[0]+oh[0])/2)*os,279+(v-(ol[1]+oh[1])/2)*os)
 d.rounded_rectangle((789,201,1080,350),12,fill=(13,21,33),outline=(54,76,99))
 d.text((803,209),'Обзор всех вокселей',font=font(14),fill=muted)
 d.polygon([op(p) for p in [(-.4,-.4,0),(.4,-.4,0),(.4,.4,0),(-.4,.4,0)]],fill=(43,59,74))
 for _,p,color,_ in items:
  ox,oy=op(p);d.rectangle((ox,oy,ox+1,oy+1),fill=color)
 b=proj(base);d.ellipse((b[0]-6,b[1]-6,b[0]+6,b[1]+6),fill=blue,outline=white,width=1)
 d.text((60,135),f'Физическое время: {t:.3f} с',font=font(22),fill=white)
 d.text((60,175),'Синий кубик 3×3×3: падение и удар о пол',font=font(17),fill=blue)
 # Gravity vector and world datum.
 a=proj((-.34,-.22,.27));b=proj((-.34,-.22,.1));d.line([a,b],fill=orange,width=4)
 d.polygon([(b[0],b[1]+8),(b[0]-7,b[1]-7),(b[0]+7,b[1]-7)],fill=orange)
 d.text((a[0]-45,a[1]-29),'g = −9,81',font=font(17),fill=orange)
 d.text((58,885),'Пол: 6 400 неподвижных вокселей · верхняя поверхность Z = 0',font=font(18),fill=muted)
 d.text((58,916),'Камера контактной зоны + общий обзор; пол имеет размер 80×80 см',font=font(16),fill=muted)
 d.rounded_rectangle((1125,111,1568,949),20,fill=(20,31,47),outline=(43,61,79),width=2)
 def txt(y,s,n=20,col=white):d.text((1150,y),s,font=font(n),fill=col)
 txt(140,'ГРАНИЧНЫЕ УСЛОВИЯ',21)
 txt(184,'Топор: свободен вдоль X',19,blue)
 txt(218,'Y / Z и поворот закреплены',19,blue)
 txt(259,'Боковой уход основания',18,muted)
 txt(289,f'{f["f"][11]*1000:.6f} мм',27)
 txt(342,'Скорость топора по X',18,muted)
 txt(372,f'{f["f"][7]:.3f} м/с',27)
 txt(424,'Растущая сила: +12 Н/с',18,muted)
 txt(454,f'{f["f"][8]:.2f} Н',27,orange)
 txt(509,f'Разорванных связей: {int(f["f"][9])}',19)
 txt(543,f'Компонент: {int(f["f"][10])}',19)
 probe=[v for v in f['v'] if v[0]>=4320];bottom=min(v[3] for v in probe)-.005
 txt(596,'Нижняя грань падающего кубика',17,muted)
 txt(627,f'Z = {bottom:.4f} м',24,blue)
 txt(687,'Это запись физического прогона.',17,muted)
 txt(714,'Движения не заданы анимацией.',17,muted)
 txt(758,'Статичный пол показан сеткой;',17,muted)
 txt(785,'в расчёте участвуют все его REV.',17,muted)
 txt(847,'Чрезмерное крошение — дефект.',16,orange)
 txt(821,'Полный переруб не проверен.',16,muted)
 txt(874,'Видео не измеряет FPS симуляции.',16,muted)
 d.text((44,963),f'Кадр решателя {n}/{steps} • шаг {dt*1000:.1f} мс • гравитация включена • ориентация топора фиксирована',font=font(16),fill=muted)
 im.save(out/f'{index:05}.png')
 stats.append({'frame':n,'broken':int(f['f'][9]),'drift_m':f['f'][11],'probe_bottom_m':bottom})
 if index%30==0:print('render',index,'/',len(ordered),flush=True)
Path(str(src)+'.render.json').write_text(json.dumps({'frames':len(ordered),'dt':dt,'seconds':steps*dt,'stats':stats},indent=2))
gif=src.with_suffix('.gif');video=src.with_suffix('.mp4')
sequence=str(out/'%05d.png')
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate','10','-i',sequence,
 '-filter_complex','[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer',str(gif)],check=True)
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate','10','-i',sequence,
 '-c:v','libx264','-pix_fmt','yuv420p','-crf','18','-movflags','+faststart',str(video)],check=True)
print('done',len(ordered),'GIF',gif,'MP4',video)
