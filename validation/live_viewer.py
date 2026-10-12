#!/usr/bin/env python3
"""Stream a solver CSV to a browser, or play a saved solver video."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import mimetypes
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "knife-cut-demo.mp4"

PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voxel Sim · GIF расчёта</title>
<style>
:root{color-scheme:dark;font:16px/1.5 system-ui,sans-serif;background:#0b1220;color:#e8eef7}
*{box-sizing:border-box}body{margin:0;padding:28px;max-width:1320px;margin-inline:auto}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:20px}
h1{font-size:clamp(22px,3vw,34px);margin:0}.pill{color:#8de0b0;border:1px solid #276b4a;border-radius:99px;padding:5px 12px;font-size:13px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:18px}.card{background:#121d2d;border:1px solid #26364c;border-radius:16px;overflow:hidden}
.gif-preview{background:#07101c;min-height:180px;display:grid;place-items:center}.gif-preview img{display:block;width:100%;height:auto;min-height:180px;max-height:75vh;object-fit:contain;background:#07101c}.gif-preview img[hidden]{display:none}.gif-error{display:none;padding:34px;text-align:center;color:#f2c17e}.gif-caption{padding:8px 18px;color:#9aabc0;font-size:13px}.live-preview{border-top:1px solid #26364c;background:#07101c}.live-preview canvas{display:block;width:100%;aspect-ratio:16/9;background:#07101c}.live-caption{padding:8px 18px;color:#9aabc0;font-size:13px}.pad{padding:18px}
.muted{color:#9aabc0}.stats{display:grid;grid-template-columns:1fr 1fr;gap:12px}.stat{background:#0c1523;border-radius:12px;padding:12px;font-size:13px}
.stat b{display:block;font-size:20px;margin-top:4px}.warn{color:#f2c17e}.source{font-size:13px;margin-top:18px;overflow-wrap:anywhere}.recording{display:none;margin-top:10px}.recording a{color:#b8d7ff}
code{color:#b8d7ff}@media(max-width:850px){body{padding:14px}.grid{grid-template-columns:1fr}}
</style></head>
<body>
<header><h1>Voxel Sim · GIF расчёта</h1><span class="pill" id="badge">Подключение…</span></header>
<main class="grid">
 <section class="card">
  <div class="gif-preview"><img id="animation" src="/gif" alt="Зацикленная GIF-анимация физического расчёта"><div class="gif-error" id="gif-error">GIF не загрузился. Проверьте, что рядом с CSV лежит файл с тем же именем и расширением .gif.</div></div>
  <div class="gif-caption">Зацикленная GIF-анимация физического расчёта · <a id="gif-link" href="/gif" target="_blank">открыть GIF отдельно</a></div>
  <div class="live-preview" id="live-preview" hidden><canvas id="live-scene"></canvas><div class="live-caption" id="live-caption">Ожидание живого кадра из CSV…</div></div>
  <div class="pad"><b id="title">Подключение к расчёту…</b><div class="muted" id="description"></div><div class="source" id="source"></div><div class="recording" id="recording"><a href="/video" target="_blank">Открыть готовую видеозапись MP4</a></div></div>
 </section>
 <aside class="card pad"><h2>Запись расчёта</h2><p class="muted">GIF воспроизводится в цикле прямо на этой странице. Для обновления анимации положите рядом с источником файл с тем же именем и расширением <code>.gif</code>.</p><div class="recording" id="recording"><a href="/video" target="_blank">Открыть MP4</a></div></aside>
</main>
<script>
const $=s=>document.querySelector(s), setText=(id,text)=>$('#'+id).textContent=text;
const animation=$('#animation');
animation.addEventListener('error',()=>{$('#gif-error').style.display='block';animation.hidden=true});
const canvas=$('#live-scene'),ctx=canvas.getContext('2d');
const live={offset:0,carry:'',dt:0,steps:0,floor:[],cells:[],frames:new Map(),latest:null,gifMtime:0};
const fmt=(v,d=3)=>Number.isFinite(v)?v.toFixed(d):'—';
function liveReset(){live.offset=0;live.carry='';live.floor=[];live.cells=[];live.frames.clear();live.latest=null}
function getFrame(n){if(!live.frames.has(n))live.frames.set(n,{n,tool:[],voxels:[]});return live.frames.get(n)}
function parseLine(line){if(!line)return;const r=line.split(',');if(r[0]==='H'){live.dt=+r[1];live.steps=+r[2]}else if(r[0]==='P')live.floor.push(r.slice(1,4).map(Number));else if(r[0]==='T')live.cells.push(r.slice(1,4).map(Number));else if(r[0]==='F'){const f=getFrame(+r[1]);f.tool=r.slice(2).map(Number);live.latest=f}else if(r[0]==='V'){getFrame(+r[1]).voxels.push({i:+r[2],p:r.slice(3,6).map(Number),damage:+r[6],lost:+r[7]})}}
function project(p,w,h,s){return[w*.5+(p[0]+p[1])*.70710678*s,h*.72+((p[0]-p[1])*.40824829-p[2]*.81649658)*s]}
function rotate(q,p){const[x,y,z,w]=q,[a,b,c]=p,tx=2*(y*c-z*b),ty=2*(z*a-x*c),tz=2*(x*b-y*a);return[a+w*tx+y*tz-z*ty,b+w*ty+z*tx-x*tz,c+w*tz+x*ty-y*tx]}
function paintLive(){const w=canvas.clientWidth,h=canvas.clientHeight,dpr=devicePixelRatio||1;if(!w||!h)return;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#07101c';ctx.fillRect(0,0,w,h);const f=live.latest;if(!f||!f.tool.length){ctx.fillStyle='#9aabc0';ctx.fillText('Ожидание первого сохранённого состояния…',20,32);return}const s=Math.min(w/1.45,h/1.05),items=[];if(live.floor.length){const xs=live.floor.map(p=>p[0]),ys=live.floor.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),poly=[[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]].map(p=>project(p,w,h,s));ctx.beginPath();poly.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fillStyle='#202f40';ctx.fill();ctx.strokeStyle='#71869c';ctx.stroke()}for(const v of f.voxels){const color=v.i>=4320?'#50b5f8':v.lost>=2?'#d94a3a':v.lost?'#f08e44':v.damage>0?'#d99a54':'#985f3b';items.push({p:v.p,c:color,z:v.p[0]-v.p[1]+v.p[2],r:v.i>=4320?4:2.6})}const q=f.tool.slice(3,7),pos=f.tool.slice(0,3);if(q.length===4)for(const cell of live.cells){const r=rotate(q,cell),p=pos.map((v,i)=>v+r[i]);items.push({p,c:'#c9d4df',z:p[0]-p[1]+p[2],r:3})}items.sort((a,b)=>a.z-b.z);for(const v of items){const[x,y]=project(v.p,w,h,s);ctx.fillStyle=v.c;ctx.fillRect(x-v.r/2,y-v.r/2,v.r,v.r)}ctx.fillStyle='#e8eef7';ctx.font='14px system-ui';ctx.fillText(`t = ${fmt(f.n*live.dt,3)} s · кадр ${f.n}/${live.steps}`,16,22);$('#live-caption').textContent=`Живой кадр решателя · разорвано связей: ${fmt(f.tool[9],0)} · скорость инструмента: ${fmt(f.tool[7])} м/с`}
async function pollLive(){try{const s=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());if(s.bytes<live.offset)liveReset();const chunk=await fetch('/capture?offset='+live.offset,{cache:'no-store'}).then(r=>r.text());live.offset+=new TextEncoder().encode(chunk).length;const parts=(live.carry+chunk).split('\n');live.carry=parts.pop()||'';parts.forEach(parseLine);paintLive();if(s.gif&&s.gif_mtime!==live.gifMtime){live.gifMtime=s.gif_mtime;const u='/gif?v='+encodeURIComponent(s.gif_mtime);animation.src=u;$('#gif-link').href=u}}catch(_){$('#live-caption').textContent='Нет соединения с CSV-потоком'}}
async function init(){
 const s=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());
 setText('title',s.label);setText('source',`${s.file} · CSV: ${Number(s.bytes).toLocaleString('ru-RU')} байт`);
 $('#badge').textContent=s.gif?'GIF готова':'GIF не найдена';$('#badge').style.color=s.gif?'#8de0b0':'#f2c17e';
 $('#description').textContent=s.mode==='csv'?'Зацикленная визуализация последней сохранённой записи физического расчёта.':'Зацикленная GIF-анимация физического прогона.';
 if(s.gif){const url='/gif?v='+encodeURIComponent(s.gif_mtime||s.gif_bytes||Date.now());animation.src=url;$('#gif-link').href=url}
 else{$('#gif-error').style.display='block';animation.hidden=true}
 if(s.video)$('#recording').style.display='block';
 if(s.mode==='csv'){$('#live-preview').hidden=false;window.addEventListener('resize',paintLive);await pollLive();setInterval(pollLive,500)}
}
init().catch(()=>{$('#badge').textContent='Ошибка подключения';setText('source','Не удалось загрузить состояние просмотрщика')});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    source_path = DEFAULT_VIDEO
    mode = "video"
    label = "Сохранённая демонстрация контакта ножа"

    @classmethod
    def video_path(cls):
        return cls.source_path if cls.mode == "video" else cls.source_path.with_suffix(".mp4")

    @classmethod
    def gif_path(cls):
        return cls.source_path.with_suffix(".gif")

    def send_bytes(self, status, content_type, payload, extra=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        route = urlparse(self.path)
        if route.path == "/":
            self.send_bytes(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            return
        if route.path == "/api/status":
            try:
                size = self.source_path.stat().st_size
            except FileNotFoundError:
                size = 0
            payload = json.dumps({
                "mode": self.mode,
                "label": self.label,
                "file": self.source_path.name,
                "bytes": size,
                "video": self.video_path().is_file(),
                "gif": self.gif_path().is_file(),
                "gif_bytes": self.gif_path().stat().st_size if self.gif_path().is_file() else 0,
                "gif_mtime": self.gif_path().stat().st_mtime_ns if self.gif_path().is_file() else 0,
            }, ensure_ascii=False).encode("utf-8")
            self.send_bytes(200, "application/json; charset=utf-8", payload)
            return
        if route.path == "/capture" and self.mode == "csv":
            try:
                offset = max(0, int(parse_qs(route.query).get("offset", ["0"])[0]))
                with self.source_path.open("rb") as source:
                    source.seek(min(offset, self.source_path.stat().st_size))
                    payload = source.read()
            except FileNotFoundError:
                payload = b""
            except (OSError, ValueError):
                self.send_error(400, "Invalid capture offset")
                return
            self.send_bytes(200, "text/plain; charset=utf-8", payload)
            return
        if route.path == "/gif":
            try:
                gif_path = self.gif_path()
                payload = gif_path.read_bytes()
                self.send_bytes(200, "image/gif", payload)
            except OSError:
                self.send_error(404, "GIF animation not found")
            return
        if route.path == "/video":
            try:
                video_path = self.video_path()
                size = video_path.stat().st_size
                content_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
                range_header = self.headers.get("Range")
                start, end = 0, size - 1
                if range_header and range_header.startswith("bytes="):
                    first, _, last = range_header[6:].partition("-")
                    if first:
                        start = int(first)
                        end = min(int(last), size - 1) if last else end
                    elif last:
                        start = max(0, size - int(last))
                    if start >= size or start > end:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                length = end - start + 1
                self.send_response(206 if range_header else 200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                if range_header:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with video_path.open("rb") as source:
                    source.seek(start)
                    remaining = length
                    while remaining and (chunk := source.read(min(64 * 1024, remaining))):
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError:
                self.send_error(404, "Video file not found")
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    source.add_argument("--csv", type=Path, help="Append-only live solver capture CSV")
    parser.add_argument("--label", default="Демонстрация физического расчёта")
    args = parser.parse_args()
    Handler.mode = "csv" if args.csv else "video"
    Handler.source_path = (args.csv or args.video).resolve()
    Handler.label = args.label
    if Handler.mode == "video" and not Handler.source_path.is_file():
        parser.error(f"video file does not exist: {Handler.source_path}")
    if Handler.mode == "csv" and not Handler.source_path.parent.is_dir():
        parser.error(f"CSV parent directory does not exist: {Handler.source_path.parent}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Viewer listening on http://127.0.0.1:{args.port} ({Handler.mode})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
