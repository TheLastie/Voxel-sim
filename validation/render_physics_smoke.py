from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
import re

W, H = 1080, 720
BG = (11, 17, 27)
CARD = (21, 31, 45)
GRID = (31, 46, 63)
WHITE = (231, 239, 248)
MUTED = (146, 164, 184)
GREEN = (91, 219, 162)
BLUE = (82, 170, 246)
ORANGE = (246, 159, 87)

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font = ImageFont.truetype(font_path, 15)
small = ImageFont.truetype(font_path, 13)
metric = ImageFont.truetype(bold_path, 17)
heading = ImageFont.truetype(bold_path, 25)
title_font = ImageFont.truetype(bold_path, 27)

report = Path('validation/physics-smoke-results.txt').read_text()
records = {line.split(' | ', 1)[0].split(' ', 1)[1]: line for line in report.splitlines()}

def capture(test, pattern):
    match = re.search(pattern, records[test])
    if not match:
        raise ValueError(f'Could not parse {test} from smoke test report')
    return match.groups()

fall_z, fall_expected, fall_error, fall_vz, fall_vz_expected = capture(
    'free_fall_100ms', r'z=([-+0-9.eE]+) m expected=([-+0-9.eE]+) m error=([-+0-9.eE]+) m vz=([-+0-9.eE]+)/([-+0-9.eE]+) m/s'
)
ground_penetration, ground_vz = capture(
    'ground_support_250ms', r'penetration=([-+0-9.eE]+) m vz=([-+0-9.eE]+) m/s'
)
collision_gap, relative_speed, momentum_error = capture(
    'head_on_ccd_20mps', r'gap=([-+0-9.eE]+) m relative_v=([-+0-9.eE]+) m/s momentum_error=([-+0-9.eE]+) kg'
)
energy_before, energy_after = capture(
    'rigid_tool_contact', r'kinetic_energy=([-+0-9.eE]+)→([-+0-9.eE]+) J'
)


def cube(draw, cx, cy, size, color, depth=12):
    x0, x1 = cx-size/2, cx+size/2
    y0, y1 = cy-size/2, cy+size/2
    dx, dy = depth, depth*0.58
    top = tuple(min(255, int(v*1.20)) for v in color)
    side = tuple(int(v*0.68) for v in color)
    draw.polygon([(x0,y0),(x0+dx,y0-dy),(x1+dx,y0-dy),(x1,y0)], fill=top, outline=WHITE)
    draw.polygon([(x1,y0),(x1+dx,y0-dy),(x1+dx,y1-dy),(x1,y1)], fill=side, outline=WHITE)
    draw.rectangle((x0,y0,x1,y1), fill=color, outline=WHITE, width=2)
    draw.line((cx,y0+5,cx,y1-5), fill=tuple(min(255,int(v*1.25)) for v in color), width=2)


def panel(draw, box, title, subtitle):
    x0,y0,x1,y1=box
    draw.rounded_rectangle(box, 16, fill=CARD, outline=GRID, width=2)
    draw.text((x0+18,y0+14),title,font=heading,fill=WHITE)
    draw.text((x0+18,y0+45),subtitle,font=small,fill=MUTED)
    draw.rounded_rectangle((x1-73,y0+16,x1-18,y0+42),13,fill=(26,70,56))
    draw.text((x1-60,y0+21),'PASS',font=small,fill=GREEN)

frames=[]
for fi in range(64):
    t=fi/63
    im=Image.new('RGB',(W,H),BG)
    d=ImageDraw.Draw(im)
    d.text((36,24),'VOXEL PHYSICS · SMOKE TESTS',font=title_font,fill=WHITE)
    d.text((38,59),'Четыре коротких сценария на production solver',font=font,fill=MUTED)
    p1=(32,102,524,392); p2=(556,102,1048,392)
    p3=(32,414,524,704); p4=(556,414,1048,704)
    panel(d,p1,'Свободное падение','100 ms · g = −9.81 m/s²')
    panel(d,p2,'Опора на землю','250 ms · куб под собственным весом')
    panel(d,p3,'Встречный удар','20 m/s · swept CCD · шаг 1 ms')
    panel(d,p4,'Контакт ножа','Жёсткий инструмент · трение выключено')

    # Ballistic body follows the same semi-implicit-Euler curve as the run.
    area=(62,164,494,319)
    for yy in range(area[1],area[3],24): d.line((area[0],yy,area[2],yy),fill=GRID)
    start_y=191
    fall_px=81*(t*t)
    cube(d,142,start_y+fall_px,34,BLUE)
    d.line((100,start_y+3,100,start_y+74),fill=MUTED,width=2)
    d.polygon([(100,start_y+82),(95,start_y+70),(105,start_y+70)],fill=MUTED)
    d.text((191,239),f'z = {fall_z} m',font=metric,fill=WHITE)
    d.text((191,266),f'ожидание {fall_expected} m',font=small,fill=MUTED)
    d.text((191,287),f'ошибка {fall_error} m · vz {fall_vz}/{fall_vz_expected}',font=small,fill=GREEN)

    # Resting cube remains exactly on the ground contact plane.
    area=(586,164,1018,319)
    for xx in range(area[0],area[2],28): d.line((xx,area[1],xx,area[3]),fill=GRID)
    ground_y=278
    d.rectangle((598,ground_y,1006,ground_y+13),fill=(75,91,107))
    cube(d,802,ground_y-18,36,(118,193,235))
    d.line((802,205,802,242),fill=ORANGE,width=3)
    d.polygon([(802,249),(796,238),(808,238)],fill=ORANGE)
    d.text((611,181),f'проникновение {ground_penetration} m',font=metric,fill=WHITE)
    d.text((611,208),f'вертикальная скорость {ground_vz} m/s',font=small,fill=GREEN)

    # High-speed pair approaches, is caught during the step, then separates.
    xmid=278
    cy=548
    if t < .50:
        q=t/.50
        left=112+q*145
        right=444-q*145
    else:
        q=(t-.50)/.50
        left=257-q*110
        right=299+q*110
    d.line((80,cy,476,cy),fill=GRID,width=1)
    d.line((xmid-104,cy,xmid+104,cy),fill=(62,78,96),width=2)
    cube(d,left,cy,40,(70,155,235))
    cube(d,right,cy,40,(231,116,83))
    if .44<t<.58:
        rad=10+int(8*abs(math.sin(fi)))
        d.ellipse((xmid-rad,cy-rad,xmid+rad,cy+rad),fill=(255,220,119))
    d.text((62,612),f'зазор {collision_gap} m',font=metric,fill=WHITE)
    d.text((62,638),f'Δp {momentum_error} kg·m/s · относительная скорость {relative_speed} m/s',font=small,fill=GREEN)

    # Tool edge reaches contact and stops with a measurable energy loss.
    wood=(826,565)
    d.line((626,wood[1]+22,988,wood[1]+22),fill=GRID,width=2)
    cube(d,wood[0],wood[1],48,(128,179,113))
    # Leftmost tool-cell face reaches the block's right face exactly at x=850.
    tool_x=920- min(t/.47,1)*62
    for row in range(3):
        for col in range(2):
            cube(d,tool_x+col*17,wood[1]-23+row*18,16,(177,187,202),depth=5)
    if t>.43:
        d.line((852,wood[1]-30,852,wood[1]+30),fill=GREEN,width=2)
    d.text((574,612),f'E: {energy_before} → {energy_after} J',font=metric,fill=WHITE)
    d.text((574,638),'пересечения граней: нет',font=small,fill=GREEN)

    d.text((38,688),'Схематичная анимация; числа — из фактического прогона physics_smoke_tests.',font=small,fill=MUTED)
    frames.append(im)

out=Path('validation/physics-smoke.gif')
frames[0].save(out,save_all=True,append_images=frames[1:],duration=55,loop=0,optimize=True)
print(f'{out.resolve()} {out.stat().st_size} bytes')
