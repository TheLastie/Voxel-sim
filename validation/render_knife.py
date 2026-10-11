#!/usr/bin/env python3
"""Render sparse solver snapshots as an isometric, individually voxelized view."""
import csv
import math
import os
import sys
from PIL import Image, ImageDraw, ImageFont

SRC = sys.argv[1] if len(sys.argv) > 1 else 'validation/knife_capture_fixed.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'validation/knife-fixed-frame.png'
MAKE_VIDEO_FRAMES = len(sys.argv) > 3 and sys.argv[3] == 'video'
CELL = 0.01
DT = 0.001
APPROACH = 0.0
schema = 1
saved_q, saved_b, saved_e = {}, {}, {}
saved_tool_cells = []
frames = {}
with open(SRC, newline='') as f:
    for row in csv.reader(f):
        if not row:
            continue
        if row[0] == 'H':
            schema, DT, APPROACH = int(row[1]), float(row[2]), float(row[6])
        elif row[0] == 'T':
            saved_tool_cells.append(tuple(map(float,row[1:4])))
        elif row[0] == 'Q':
            x,y,z,w = map(float,row[2:6])
            saved_q[int(row[1])] = (w,x,y,z)
        elif row[0] == 'B':
            saved_b.setdefault(int(row[1]),[]).append(tuple(map(int,row[2:4])))
        elif row[0] == 'E':
            saved_e[int(row[1])] = tuple(map(float,row[2:5]))
        elif row[0] == 'F':
            frame = int(row[1])
            frames[frame] = {
                'tool': tuple(map(float, row[2:5])),
                'vel': tuple(map(float, row[5:8])),
                'omega': tuple(map(float, row[8:11])),
                'broken': int(row[11]), 'clusters': int(row[12]),
                'vmax': float(row[13]), 'wood_ke':float(row[14]), 'tool_ke':float(row[15]), 'vox': [],
            }
        elif row[0] == 'V':
            frame = int(row[1])
            if frame in frames:
                frames[frame]['vox'].append((
                    int(row[2]), *map(float, row[3:6]), float(row[6]), int(row[7])
                ))
if not frames:
    raise SystemExit('No complete frames in capture')

# Rebuild the rigid blade's saved orientation from the initial fixture rotation
# and recorded angular velocity. The old image drew a world-aligned rectangle,
# ignoring both the voxel mesh and this rotation.
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw-ax*bx-ay*by-az*bz,
            aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw)

def rotate(q, p):
    v = (0.0, *p)
    qi = (q[0], -q[1], -q[2], -q[3])
    return qmul(qmul(q, v), qi)[1:]

q0 = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
pitch_deg = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
pitch = (math.cos(math.radians(pitch_deg) * 0.5),
         0.0, math.sin(math.radians(pitch_deg) * 0.5), 0.0)
q0 = qmul(pitch, q0)
ordered_frames = sorted(frames)
frames[ordered_frames[0]]['rotation'] = q0
for prev, cur in zip(ordered_frames, ordered_frames[1:]):
    q = frames[prev]['rotation']
    omega = frames[prev]['omega']
    dt = (cur - prev) * 0.001
    mag = math.sqrt(sum(v*v for v in omega))
    if mag > 1e-10:
        half = mag * dt * 0.5
        s = math.sin(half) / mag
        dq = (math.cos(half), omega[0]*s, omega[1]*s, omega[2]*s)
        q = qmul(dq, q)
        norm = math.sqrt(sum(v*v for v in q))
        q = tuple(v / norm for v in q)
    frames[cur]['rotation'] = q
if saved_q:
    for n in ordered_frames:
        frames[n]['rotation'] = saved_q[n]

W, H = 1440, 900
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_path, 19)
small = ImageFont.truetype(font_path, 15)
title = ImageFont.truetype(font_path, 28)
mini = ImageFont.truetype(font_path, 13)
SQRT_HALF = math.sqrt(0.5)
SQRT_2_3 = math.sqrt(2.0 / 3.0)
SQRT_1_6 = math.sqrt(1.0 / 6.0)
STAGE = (60, 120, 910, 825)

def iso(p):
    """Equal-scale isometric projection; p is in REV units."""
    x, y, z = p
    return ((x + y) * SQRT_HALF,
            (x - y) * SQRT_1_6 - z * SQRT_2_3)

def blade_local_cells():
    # Match Shape::Blade in Tool::new: 20 x 5 x 14, tapering to a one-REV edge.
    local_cells = []
    for ix in range(20):
        for iy in range(5):
            for iz in range(14):
                thickness = 0.5 + (19 - ix) / 19 * 2.0
                local_y = (iy + 0.5 - 2.5) * CELL
                if abs(local_y / CELL) > thickness:
                    continue
                local = ((ix + 0.5 - 10.0) * CELL,
                         local_y,
                         (iz + 0.5 - 7.0) * CELL)
                local_cells.append(local)
    centroid = tuple(sum(p[k] for p in local_cells)/len(local_cells) for k in range(3))
    return local_cells, centroid

LOCAL_BLADE, BLADE_CENTROID = blade_local_cells()
if saved_tool_cells:
    LOCAL_BLADE, BLADE_CENTROID = saved_tool_cells, (0.0,0.0,0.0)

def blade_cells(fr):
    q = fr['rotation']
    center = fr['tool']  # Tool position now tracks its occupied-volume COM.
    for local in LOCAL_BLADE:
        # Tool::new preserves the original shape while shifting its local
        # origin to the center of mass. Reapply that same centroid here.
        offset = rotate(q, tuple(local[k]-BLADE_CENTROID[k] for k in range(3)))
        yield tuple((center[k] + offset[k]) / CELL for k in range(3))

def cube_faces(center, half, project, rotation=None):
    x, y, z = center
    a = half
    def face(points):
        offsets = [(px*a,py*a,pz*a) for px,py,pz in points]
        if rotation is not None:
            offsets = [rotate(rotation,p) for p in offsets]
        return [project((x+px,y+py,z+pz)) for px,py,pz in offsets]
    if rotation is not None:
        candidates=[
            ((1,0,0),[(1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)]),
            ((-1,0,0),[(-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,1,-1)]),
            ((0,-1,0),[(-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1)]),
            ((0,1,0),[(-1,1,-1),(-1,1,1),(1,1,1),(1,1,-1)]),
            ((0,0,1),[(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]),
            ((0,0,-1),[(-1,-1,-1),(-1,1,-1),(1,1,-1),(1,-1,-1)])]
        return [face(points) for normal,points in candidates
                if sum(a*b for a,b in zip(rotate(rotation,normal),(1,-1,1)))>1e-6]
    # Camera direction sees the +X, -Y and +Z faces.
    return [
        face([(1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)]),
        face([(-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1)]),
        face([(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]),
    ]

# Use one camera for the whole clip. Reframing each snapshot made the specimen
# appear to drift and change size while the knife moved.
camera_points = []
for n,fr in frames.items():
    # Close camera covers the original specimen and the contact region. The
    # complete metre-scale approach is represented by the X overview strip.
    if schema >= 2 and n*DT < APPROACH:
        continue
    wood_centers = [(x / CELL, y / CELL, z / CELL)
                    for _, x, y, z, _, _ in fr['vox']]
    camera_points.extend(iso(p) for p in wood_centers)
    camera_points.extend(iso(p) for p in blade_cells(fr))
if not camera_points:
    camera_points.extend(iso((x/CELL,y/CELL,z/CELL)) for _,x,y,z,_,_ in frames[ordered_frames[0]]['vox'])
overview_min = min([fr['tool'][0] for fr in frames.values()]+[-0.15])
overview_max = max([fr['tool'][0] for fr in frames.values()]+[0.15])
min_x = min(p[0] for p in camera_points) - 1
max_x = max(p[0] for p in camera_points) + 1
min_y = min(p[1] for p in camera_points) - 1
max_y = max(p[1] for p in camera_points) + 1
scale = min((STAGE[2]-STAGE[0])/(max_x-min_x),
            (STAGE[3]-STAGE[1])/(max_y-min_y))
cx = (STAGE[0]+STAGE[2])*0.5
cy = (STAGE[1]+STAGE[3])*0.5
pcx = (min_x+max_x)*0.5
pcy = (min_y+max_y)*0.5
def project(p):
    sx, sy = iso(p)
    return (cx+(sx-pcx)*scale, cy+(sy-pcy)*scale)

def render(n, dst):
    fr = frames[n]
    wood = fr['vox']
    blade = list(blade_cells(fr))
    wood_centers = [(x/CELL, y/CELL, z/CELL) for _,x,y,z,_,_ in wood]
    im = Image.new('RGB', (W,H), (13,19,28))
    d = ImageDraw.Draw(im)
    tool_mask = Image.new('L',(W,H),0)
    tool_mask_draw = ImageDraw.Draw(tool_mask)
    d.text((42,25), 'VOXEL PHYSICS LAB  /  CONTACT DIAGNOSTIC', font=title, fill=(237,242,248))
    d.text((44,65), f'1 cm REV  •  {max(frames)*DT:.3f} s physical run  •  Exact poses / bonds' if schema>=2 else '1 cm cells shown individually  •  Saved solver states  •  20 ms physical run',
           font=small, fill=(145,163,184))
    d.rounded_rectangle(STAGE, radius=16, fill=(17,25,36), outline=(39,52,68), width=2)

    items = []
    for voxel, center in zip(wood, wood_centers):
        i, x, y, z, damage, lost = voxel
        # lost is an aggregate of missing positive-axis neighbor bonds; remove
        # the specimen boundaries from the count before coloring.
        gx, gy, gz = i%12, (i//12)%12, i//144
        lost = max(0, lost-(gx==11)-(gy==11)-(gz==29))
        if lost >= 2:
            base = (212,72,62)
        elif lost == 1:
            base = (235,133,62)
        elif damage > .35:
            base = (239,190,76)
        elif damage > .08:
            base = (189,158,85)
        else:
            base = (113,153,111)
        depth = center[0] - center[1] + center[2]
        items.append((depth, 0, center, base))
    for center in blade:
        depth = center[0] - center[1] + center[2]
        items.append((depth, 1, center, (177,196,218)))

    # Separate each 1 cm REV slightly so its cubic outline remains visible.
    for _, is_blade, center, color in sorted(items, key=lambda item: item[0]):
        pc = project(center)
        if schema>=2 and not (STAGE[0]+10 <= pc[0] <= STAGE[2]-10 and STAGE[1]+10 <= pc[1] <= STAGE[3]-10):
            continue
        faces = cube_faces(center, 0.5 if is_blade else 0.49, project,
                           fr['rotation'] if is_blade else None)
        if is_blade:
            shades = [(146,166,192), (171,190,213), (221,230,240)]
            outline = (45,59,76)
        else:
            shades = [tuple(int(c*.72) for c in color),
                      tuple(int(c*.86) for c in color), color]
            outline = (29,42,52)
        for polygon, shade in zip(faces, shades):
            d.polygon(polygon, fill=shade, outline=outline)
            if is_blade:
                tool_mask_draw.polygon(polygon,fill=255)

    # Mark actual gaps between originally adjacent cells. The CSV stores lost
    # bond counts, not per-axis flags, so require both a lost bond and a visible
    # center-to-center gap before drawing the red fracture trace.
    by_id = {}
    for voxel, center in zip(wood, wood_centers):
        i, _, _, _, _, raw_lost = voxel
        gx, gy, gz = i%12, (i//12)%12, i//144
        outside = int(gx==11) + int(gy==11) + int(gz==29)
        by_id[i] = (center, max(0, raw_lost-outside))
    if schema>=2:
        for i,j in saved_b.get(n,[]):
            if i not in by_id or j not in by_id:
                continue
            a,b = by_id[i][0],by_id[j][0]
            gap = math.dist(a,b)
            pa,pb = project(a),project(b)
            probes=[pa,pb,((pa[0]+pb[0])*.5,(pa[1]+pb[1])*.5)]
            blade_occludes = any(tool_mask.getpixel((max(0,min(W-1,round(p[0]))),max(0,min(H-1,round(p[1]))))) for p in probes)
            if not blade_occludes and 1.02 < gap < 2.0 and all(STAGE[0]<p[0]<STAGE[2] and STAGE[1]<p[1]<STAGE[3] for p in [pa,pb]):
                d.line([pa,pb],fill=(239,74,77),width=2)
    for i,(a,lost_a) in ([] if schema>=2 else by_id.items()):
        gx = i % 12
        # The knife advances through world X, so the cut surface is traced by
        # missing X-neighbor ties only; showing all axes looked like a red cage.
        j = i + 1
        if gx == 11 or j not in by_id:
            continue
        b,lost_b = by_id[j]
        gap = math.sqrt(sum((b[k]-a[k])**2 for k in range(3)))
        if lost_a+lost_b and 1.01 < gap < 1.5:
            d.line([project(a),project(b)], fill=(239,74,77), width=3)

    # Minimal isometric axes make the camera orientation easy to read.
    ax_origin = project((-7,-7,0))
    for end, label, color in [((-5,-7,0),'X',(223,128,111)),
                              ((-7,-5,0),'Y',(115,181,151)),
                              ((-7,-7,2),'Z',(119,163,230))]:
        p_end = project(end)
        d.line([ax_origin,p_end],fill=color,width=3)
        d.text((p_end[0]+5,p_end[1]-7),label,font=mini,fill=color)

    bx, by = 950, 140
    d.rounded_rectangle((bx,by,1398,490),radius=18,fill=(25,34,47),outline=(49,65,83),width=2)
    d.text((bx+24,by+22),'SIMULATION STATE',font=small,fill=(140,166,193))
    rows = [('Frame',f'{n:04d} / {max(frames):04d}'),
            ('Simulation time',f'{n*DT:.3f} s'),
            ('Tool speed',f'{math.sqrt(sum(v*v for v in fr["vel"])):.3f} m/s'),
            ('Broken bonds',f'{fr["broken"]:,}'),
            ('Connected pieces',str(fr['clusters'])),
            ('Peak voxel speed',f'{fr["vmax"]:.2f} m/s'),
            ('Wood voxels',f'{len(wood):,} × 1 cm'),
            ('Knife voxels',f'{len(blade):,} × 1 cm')]
    if schema>=2:
        rows=rows[:6]+[('Tool / wood KE',f'{fr["tool_ke"]:.2f} / {fr["wood_ke"]:.2f} J'),('Elastic energy',f'{saved_e[n][1]:.3f} J')]
    for j,(label,value) in enumerate(rows):
        yy=by+62+j*37
        d.text((bx+24,yy),label,font=small,fill=(161,177,196))
        d.text((bx+240,yy),value,font=small,fill=(238,242,247))

    d.text((bx,535),'Bond / material state',font=small,fill=(164,180,199))
    legend=[((113,153,111),'Intact bonds'),((239,190,76),'Damaged bonds'),((235,133,62),'Multiple broken bonds'),((239,74,77),'Visible local separation'),((177,196,218),'Steel REV')]
    for j,(color,label) in enumerate(legend):
        yy=565+j*30
        d.rectangle((bx,yy,bx+18,yy+18),fill=color,outline=(35,47,60))
        d.text((bx+29,yy-2),label,font=small,fill=(185,197,211))
    d.text((bx,735),'FIXTURE CHECKS',font=small,fill=(140,166,193))
    checks=[('Tension / fiber ratio','39.46×'),('Compression / fiber ratio','31.12×'),('Bending','9 / 9 bonds'),('Wedge','264 breaks; 1 piece')]
    for j,(label,value) in enumerate(checks):
        yy=762+j*24
        d.text((bx,yy),label,font=mini,fill=(161,177,196))
        d.text((bx+260,yy),value,font=mini,fill=(238,190,115) if j==3 else (238,242,247))
    if schema>=2:
        d.rectangle((bx,728,1439,850),fill=(13,19,28))
        d.text((bx,735),'FULL APPROACH / X POSITION',font=small,fill=(140,166,193))
        def strip(x): return bx+12+(x-overview_min)/(overview_max-overview_min)*420
        sy=790
        d.line([(bx+12,sy),(bx+432,sy)],fill=(88,108,129),width=2)
        wx=strip(0.0)
        d.rectangle((wx-3,sy-14,wx+3,sy+14),fill=(113,153,111))
        tx=strip(fr['tool'][0])
        d.ellipse((tx-5,sy-5,tx+5,sy+5),fill=(221,230,240))
        d.text((bx,815),f'Wood x=0 m    Knife x={fr["tool"][0]:.3f} m',font=mini,fill=(161,177,196))
    footer = 'Recorded solver states. Gravity disabled. Cracks shown only for saved broken neighbor bonds.' if schema>=2 else 'Diagnostic only: many bonds break, but the wood remains one connected piece.'
    d.text((42,865),footer,
           font=mini,fill=(112,130,151))
    im.save(dst)
    return im

selected = int(sys.argv[4]) if len(sys.argv)>4 else max(frames)
if selected not in frames:
    raise SystemExit(f'Frame {selected} was not captured')
render(selected, OUT)
print(f'{len(frames)} frames loaded; preview frame {selected}={OUT}')
if MAKE_VIDEO_FRAMES:
    frame_dir = os.path.splitext(OUT)[0] + '_frames'
    os.makedirs(frame_dir, exist_ok=True)
    for frame in sorted(frames):
        render(frame, f'{frame_dir}/{frame:04d}.png')
