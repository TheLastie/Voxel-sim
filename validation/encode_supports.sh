#!/usr/bin/env bash
set -euo pipefail
ffmpeg -v error -threads 2 -framerate 20 -i validation/supports_frames/%05d.png -vf 'tpad=stop_mode=clone:stop_duration=2,format=yuv420p' -r 30 -c:v libx264 -preset medium -crf 17 -movflags +faststart /workspace/scratch/2cecc6c5f3ec/mounted-axe-floor.mp4
ffmpeg -v error -threads 2 -i /workspace/scratch/2cecc6c5f3ec/mounted-axe-floor.mp4 -filter_complex 'fps=20,scale=1280:800:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=bayer:bayer_scale=4' /workspace/scratch/2cecc6c5f3ec/mounted-axe-floor.gif
python validation/verify_supports_video.py /workspace/scratch/2cecc6c5f3ec/mounted-axe-floor.mp4 /workspace/scratch/2cecc6c5f3ec/mounted-axe-floor.gif validation/supports_frames
python - <<'PY'
import gzip,shutil
from pathlib import Path
with Path('validation/supports.csv').open('rb') as f,gzip.open('/workspace/scratch/2cecc6c5f3ec/mounted-axe-floor-states.csv.gz','wb',compresslevel=6) as g:shutil.copyfileobj(f,g)
PY
