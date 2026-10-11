#!/usr/bin/env python3
import json,subprocess,sys
from pathlib import Path
from PIL import Image
video=Path(sys.argv[1]); gif=Path(sys.argv[2]); frames=Path(sys.argv[3])
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
s=next(s for s in probe['streams'] if s['codec_type']=='video')
assert (s['width'],s['height'])==(1600,1000)
assert s['pix_fmt']=='yuv420p'
assert len(list(frames.glob('*.png')))==151
assert float(probe['format']['duration'])>9.4
subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
im=Image.open(gif)
for i in range(im.n_frames):im.seek(i);im.load()
print(json.dumps({'resolution':[1600,1000],'duration_s':float(probe['format']['duration']),'mp4_frames':int(s['nb_frames']),'gif_frames':im.n_frames,'source_states':151,'decode':'pass'}))
