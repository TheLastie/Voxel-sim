#!/usr/bin/env python3
"""Render the physics-tier demonstration recorded by `tier_demo_capture`.

Every frame is a recorded solver state (no interpolation). Colour shows the tier each REV
runs at; the right panel shows the measured wall time of `solver::step` (mean of the 10
steps since the previous frame) and the tier census.
Usage: render_tier_demo.py validation/tier-demo.csv validation/tier-demo.gif
"""
import csv, math, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

src = Path(sys.argv[1] if len(sys.argv) > 1 else "validation/tier-demo.csv")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else src.with_suffix(".gif"))
frames, header, totals = {}, None, None
for row in csv.reader(src.open(newline="")):
    if not row:
        continue
    if row[0] == "H":
        header = (float(row[1]), int(row[2]), int(row[3]), int(row[4]))
    elif row[0] == "S":
        frames[int(row[1])] = {"ms": float(row[2]), "census": list(map(int, row[3:8])), "v": []}
    elif row[0] == "V":
        frames[int(row[1])]["v"].append((float(row[3]), float(row[4]), float(row[5]), int(row[6])))
    elif row[0] == "T":
        totals = tuple(map(float, row[1:4]))
dt, steps, every, release = header
if len(frames) < 176:
    raise SystemExit(f"need 176+ recorded states, got {len(frames)}")

def font(size):
    for name in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

W, H = 1600, 1000
BG, WHITE, MUTED = (9, 16, 27), (232, 239, 247), (140, 160, 184)
TIER = {0: ("спит (T0)", (88, 140, 230)), 1: ("закреплён", (92, 98, 108)), 2: ("жёсткий (T1)", (190, 120, 220)),
        3: ("балка (T2)", (90, 200, 120)), 4: ("ячейки (T3)", (240, 150, 60))}
S = 900.0  # pixels per metre
cos30, sin30 = math.cos(math.radians(30)), math.sin(math.radians(30))
def project(x, y, z):
    return (560 + (x - y) * cos30 * S, 640 + (x + y) * sin30 * S - z * S)

ms_series = [(step, f["ms"]) for step, f in sorted(frames.items()) if step > 0]
images = []
for n, (step, f) in enumerate(sorted(frames.items()), 1):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((40, 26), "УРОВНИ ФИЗИКИ В ОДНОЙ СЦЕНЕ · сон, балка, пробуждение от удара", font=font(28), fill=WHITE)
    d.text((42, 70), f"t = {step*dt:5.2f} с · кадр {n}/{len(frames)} · шаг {step}/{steps} · реальные состояния решателя, без интерполяции",
           font=font(18), fill=MUTED)
    half = 0.0045 * S
    for x, y, z, tier in sorted(f["v"], key=lambda v: (v[0] + v[1]) - v[2] * 0.01):
        cx, cy = project(x, y, z)
        colour = TIER[tier][1]
        top = [(cx, cy - half), (cx + half * cos30 * 2, cy), (cx, cy + half), (cx - half * cos30 * 2, cy)]
        d.polygon(top, fill=colour, outline=tuple(min(255, c + 40) for c in colour))
        d.polygon([top[3], top[2], (cx, cy + 2.6 * half), (top[3][0], top[3][1] + 1.6 * half)],
                  fill=tuple(int(c * 0.6) for c in colour))
        d.polygon([top[2], top[1], (top[1][0], top[1][1] + 1.6 * half), (cx, cy + 2.6 * half)],
                  fill=tuple(int(c * 0.8) for c in colour))
    # Right panel.
    px = 1130
    d.text((px, 130), "Уровни", font=font(22), fill=WHITE)
    for k, (tier, (name, colour)) in enumerate(TIER.items()):
        d.rectangle((px, 172 + k * 34, px + 22, 194 + k * 34), fill=colour)
        d.text((px + 34, 170 + k * 34), f"{name}: {f['census'][tier]}", font=font(19), fill=WHITE)
    d.text((px, 360), "Стоимость шага 10 мс (стена, 16 потоков)", font=font(19), fill=WHITE)
    d.text((px, 388), f"сейчас: {f['ms']:.2f} мс", font=font(30), fill=(255, 220, 120))
    # Log-scale history of step cost up to this frame.
    gx, gy, gw, gh = px, 450, 420, 220
    d.rectangle((gx, gy, gx + gw, gy + gh), outline=(60, 76, 96))
    lo, hi = math.log10(0.05), math.log10(1000.0)
    for ref in (0.1, 1, 5, 10, 100, 1000):
        yy = gy + gh - (math.log10(ref) - lo) / (hi - lo) * gh
        d.line((gx, yy, gx + gw, yy), fill=(34, 46, 62) if ref != 5 else (120, 60, 60))
        d.text((gx - 52, yy - 10), f"{ref:g}", font=font(14), fill=MUTED)
    d.text((gx + gw - 140, gy + gh - (math.log10(5) - lo) / (hi - lo) * gh - 22), "цель 5 мс", font=font(14), fill=(200, 110, 110))
    pts = [(gx + s / steps * gw, gy + gh - (math.log10(max(m, 0.05)) - lo) / (hi - lo) * gh) for s, m in ms_series if s <= step]
    if len(pts) > 1:
        d.line(pts, fill=(255, 220, 120), width=2)
    rx = gx + release / steps * gw
    d.line((rx, gy, rx, gy + gh), fill=(120, 200, 255))
    d.text((rx + 4, gy + 4), "шар отпущен", font=font(14), fill=(120, 200, 255))
    d.text((gx, gy + gh + 8), "мс на шаг (лог. шкала) по времени сцены", font=font(15), fill=MUTED)
    notes = [
        "Что видно в этом прогоне (честно):",
        "+ блок засыпает, консоль регулятор перевёл в балку:",
        "  шаг 85 → 1,2 мс до t = 4 с (цель 5 мс);",
        "+ падающий шар будит блок за ~9 см до касания;",
        "− после удара тела засыпают и тут же просыпаются:",
        "  тепло удара будит их (цикл), шаг ~200 мс.",
    ]
    for k, line in enumerate(notes):
        d.text((px, 730 + k * 26), line, font=font(16), fill=MUTED)
    if totals:
        d.text((40, H - 40), f"Весь прогон: p50 {totals[0]:.2f} мс · p95 {totals[1]:.1f} мс · max {totals[2]:.0f} мс на шаг · i5-14450HX, только решатель",
               font=font(16), fill=MUTED)
    images.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
images[0].save(dst, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
print(f"wrote {dst}: {len(images)} frames at 10 fps = {len(images)/10:.1f} s")
