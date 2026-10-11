"""Encode captured PNGs with an explicitly labelled physical-time mapping."""
import csv
import sys
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

prefix = sys.argv[1] if len(sys.argv) > 1 else 'knife-long'
rows = list(csv.reader(Path(f'validation/{prefix}.csv').open()))
frames = sorted(int(r[1]) for r in rows if r[0] == 'F')
durations = []
slow_motion = []
base_slowdown = 2 if prefix in ('knife-ramp', 'knife-gentle') else 1
for a, b in zip(frames, frames[1:]):
    dt = (b-a)*0.001
    overlap = max(0.0, min(b*0.001, 1.24)-max(a*0.001, 0.96))
    durations.append((dt-overlap)*base_slowdown+overlap*20)
    slow_motion.append(overlap > 0)
durations.append(0.04)
slow_motion.append(False)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
images = []
paths = []
for index, n in enumerate(frames):
    path = Path(f'validation/{prefix}-final_frames') / f'{n:04d}.png'
    im = Image.open(path).convert('RGB')
    label = 'CONTACT REPLAY: SLOW MOTION' if slow_motion[index] else ('PLAYBACK: 2x SLOWER' if base_slowdown == 2 else 'PLAYBACK: REAL PHYSICAL TIME')
    ImageDraw.Draw(im).text((60, 95), label, font=font, fill=(220, 172, 100))
    im.save(path)
    paths.append(path.resolve())
    images.append(im.resize((1152, 720), Image.Resampling.LANCZOS).quantize(colors=192, method=Image.Quantize.FASTOCTREE))
concat = ''.join(f"file '{path}'\noption framerate 50\nduration {t:.6f}\n" for path, t in zip(paths, durations))
concat += f"file '{paths[-1]}'\noption framerate 50\n"
Path(f'validation/{prefix}-concat.txt').write_text(concat)
images[0].save(f'../{prefix}-demo.gif', save_all=True, append_images=images[1:],
               duration=[max(10, round(t*1000/10)*10) for t in durations], loop=0, optimize=True)
subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                '-i', f'validation/{prefix}-concat.txt', '-fps_mode', 'vfr', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', f'../{prefix}-demo.mp4'], check=True)
Path(f'validation/{prefix}-timing.json').write_text(json.dumps({
    'physical_seconds': max(frames)*0.001, 'playback_seconds_planned': sum(durations),
    'slowdown_interval': [0.96, 1.24], 'slowdown_factor': 20,
    'base_slowdown_factor': base_slowdown,
    'frame_count': len(frames)}, indent=2)+'\n')
print('Playback seconds:', sum(durations))
