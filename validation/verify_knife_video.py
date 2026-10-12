"""Check complete decoding and timing against the saved physical snapshots."""
import csv
import json
import subprocess
import sys
from pathlib import Path
from PIL import Image

prefix = sys.argv[1]
source = Path('validation') / f'{prefix}.csv'
timing = json.loads(source.with_name(f'{prefix}-timing.json').read_text())
frames = [int(r[1]) for r in csv.reader(source.open()) if r and r[0] == 'F']
video = Path('..') / f'{prefix}-demo.mp4'
gif = Path('..') / f'{prefix}-demo.gif'
probe = json.loads(subprocess.check_output([
    'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height,nb_read_frames,codec_name,pix_fmt:format=duration,size',
    '-of', 'json', str(video)], text=True))
stream = probe['streams'][0]
assert (stream['width'], stream['height']) == (1440, 900)
assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
assert int(stream['nb_read_frames']) == len(frames)+1
duration = float(probe['format']['duration'])
assert abs(duration-timing['playback_seconds_planned']) < 0.05
subprocess.run(['ffmpeg', '-xerror', '-v', 'error', '-i', str(video),
                '-f', 'null', '-'], check=True, capture_output=True)
gif_duration = 0
with Image.open(gif) as image:
    assert image.n_frames == len(frames)
    gif_frames = image.n_frames
    for n in range(image.n_frames):
        image.seek(n)
        image.load()
        gif_duration += image.info.get('duration', 0)
assert abs(gif_duration/1000-timing['playback_seconds_planned']) < 0.05
result = {'full_decode_passed': True, 'physical_seconds': timing['physical_seconds'],
          'saved_states': len(frames), 'first_state': min(frames), 'last_state': max(frames),
          'mp4_frames_including_final_hold': int(stream['nb_read_frames']),
          'mp4_duration_seconds': duration, 'gif_frames': gif_frames,
          'gif_duration_seconds': gif_duration/1000, 'resolution': [1440,900],
          'codec': stream['codec_name'], 'pixel_format': stream['pix_fmt']}
source.with_name(f'{prefix}-video-check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
