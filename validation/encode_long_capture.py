"""Encode captured PNGs with an explicitly labelled physical-time mapping."""
import csv
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

frames = sorted(int(r[1]) for r in csv.reader(Path('validation/knife-long.csv').open()) if r[0] == 'F')
durations = []
for a, b in zip(frames, frames[1:]):
    dt = (b-a)*0.001
    factor = 20 if 0.96 <= (a+b)*0.0005 <= 1.24 else 1
    durations.append(dt*factor)
durations.append(0.04)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
images = []
paths = []
for n in frames:
    path = Path('validation/knife-long-final_frames') / f'{n:04d}.png'
    im = Image.open(path).convert('RGB')
    label = 'CONTACT REPLAY: 20x SLOWER' if 0.96 <= n*0.001 <= 1.24 else 'PLAYBACK: REAL PHYSICAL TIME'
    ImageDraw.Draw(im).text((60, 95), label, font=font, fill=(220, 172, 100))
    im.save(path)
    paths.append(path.resolve())
    images.append(im.resize((1152, 720), Image.Resampling.LANCZOS).quantize(colors=192, method=Image.Quantize.FASTOCTREE))
concat = ''.join(f"file '{path}'\nduration {t:.6f}\n" for path, t in zip(paths, durations))
concat += f"file '{paths[-1]}'\n"
Path('validation/knife-long-concat.txt').write_text(concat)
images[0].save('../knife-long-demo.gif', save_all=True, append_images=images[1:],
               duration=[max(10, round(t*1000/10)*10) for t in durations], loop=0, optimize=True)
subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                '-i', 'validation/knife-long-concat.txt', '-fps_mode', 'vfr', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', '../knife-long-demo.mp4'], check=True)
Path('validation/knife-long-timing.json').write_text(json.dumps({
    'physical_seconds': 2.1, 'playback_seconds_planned': sum(durations),
    'slowdown_interval': [0.96, 1.24], 'slowdown_factor': 20,
    'frame_count': len(frames)}, indent=2)+'\n')
print('Playback seconds:', sum(durations))
