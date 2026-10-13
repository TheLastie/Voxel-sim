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
header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:20px}header select{background:#121d2d;color:#e8eef7;border:1px solid #42556d;border-radius:8px;padding:9px 12px;font:inherit}
h1{font-size:clamp(22px,3vw,34px);margin:0}.pill{color:#8de0b0;border:1px solid #276b4a;border-radius:99px;padding:5px 12px;font-size:13px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:18px}.card{background:#121d2d;border:1px solid #26364c;border-radius:16px;overflow:hidden}
.gif-preview{background:#07101c;min-height:180px;display:grid;place-items:center}.gif-preview img{display:block;width:100%;height:auto;min-height:180px;max-height:75vh;object-fit:contain;background:#07101c}.gif-preview img[hidden]{display:none}.gif-error{display:none;padding:34px;text-align:center;color:#f2c17e}.gif-caption{padding:8px 18px;color:#9aabc0;font-size:13px}.live-preview{border-top:1px solid #26364c;background:#07101c}.live-preview canvas{display:block;width:100%;aspect-ratio:16/9;background:#07101c}.live-caption{padding:8px 18px;color:#9aabc0;font-size:13px}.pad{padding:18px}
.muted{color:#9aabc0}.stats{display:grid;grid-template-columns:1fr 1fr;gap:12px}.stat{background:#0c1523;border-radius:12px;padding:12px;font-size:13px}
.stat b{display:block;font-size:20px;margin-top:4px}.warn{color:#f2c17e}.source{font-size:13px;margin-top:18px;overflow-wrap:anywhere}.recording{display:none;margin-top:10px}.recording a{color:#b8d7ff}
code{color:#b8d7ff}@media(max-width:850px){body{padding:14px}.grid{grid-template-columns:1fr}}
</style></head>
<body>
<header><h1>Voxel Sim · эксперименты</h1><select id="experiment" aria-label="Выбрать эксперимент" hidden></select><span class="pill" id="badge">Подключение…</span></header>
<main class="grid">
 <section class="card">
  <div class="gif-preview"><img id="animation" src="/gif" alt="Зацикленная GIF-анимация физического расчёта"><div class="gif-error" id="gif-error">GIF не загрузился. Проверьте, что рядом с CSV лежит файл с тем же именем и расширением .gif.</div></div>
  <div class="gif-caption">Зацикленная GIF-анимация физического расчёта · <a id="gif-link" href="/gif" target="_blank">открыть GIF отдельно</a></div>
  <div class="live-preview" id="live-preview" hidden><canvas id="live-scene"></canvas><div class="live-caption" id="live-caption">Ожидание живого кадра из CSV…</div></div>
  <div class="pad"><b id="title">Подключение к расчёту…</b><div class="muted" id="description"></div><div class="source" id="source"></div><div class="recording" id="recording-main"><a href="/video" target="_blank">Открыть готовую видеозапись MP4</a></div></div>
 </section>
 <aside class="card pad"><h2>Запись расчёта</h2><p class="muted">GIF воспроизводится в цикле прямо на этой странице. Для обновления анимации положите рядом с источником файл с тем же именем и расширением <code>.gif</code>.</p><div class="recording" id="recording-aside"><a href="/video" target="_blank">Открыть MP4</a></div></aside>
</main>
<script>
const $=s=>document.querySelector(s), setText=(id,text)=>$('#'+id).textContent=text;
const animation=$('#animation');
animation.addEventListener('error',()=>{$('#gif-error').style.display='block';animation.hidden=true});
const canvas=$('#live-scene'),ctx=canvas.getContext('2d');
const live={offset:0,carry:'',dt:0,steps:0,mode:'mechanical',floor:[],cells:[],bounds:null,events:{},frames:new Map(),latest:null,gifMtime:0,experiment:new URLSearchParams(location.search).get('experiment')||''};
const fmt=(v,d=3)=>Number.isFinite(v)?v.toFixed(d):'—';
function liveReset(){live.offset=0;live.carry='';live.floor=[];live.cells=[];live.bounds=null;live.events={};live.frames.clear();live.latest=null;live.mode='mechanical'}
function getFrame(n){if(!live.frames.has(n))live.frames.set(n,{n,tool:[],voxels:[],bonds:[],ledger:null});return live.frames.get(n)}
function parseLine(line){if(!line)return;const r=line.split(',');if(r[0]==='H'){live.dt=+r[1];live.steps=+r[2];live.mode=r[4]||'mechanical'}else if(r[0]==='P')live.floor.push(r.slice(1,4).map(Number));else if(r[0]==='T')live.cells.push(r.slice(1,4).map(Number));else if(r[0]==='W')live.bounds={min:r.slice(1,4).map(Number),max:r.slice(4,7).map(Number)};else if(r[0]==='E')live.events[r[3]]=(live.events[r[3]]||0)+1;else if(r[0]==='S')live.latest=getFrame(+r[1]);else if(r[0]==='Q'){const f=getFrame(+r[1]);f.voxels.push({i:+r[2],p:r.slice(3,6).map(Number),temp:+r[6],liquid:+r[7]||0});live.latest=f}else if(r[0]==='L'){const f=getFrame(+r[1]);f.ledger={remaining:+r[2],removed:+r[3],mass:+r[4],internal:+r[5],kinetic:+r[6],pending:+r[7],momentum:r.slice(8,11).map(Number)};live.latest=f}else if(r[0]==='B'){getFrame(+r[1]).bonds.push([+r[2],+r[3]]);live.latest=getFrame(+r[1])}else if(r[0]==='F'){const f=getFrame(+r[1]);f.tool=r.slice(2).map(Number);live.latest=f}else if(r[0]==='V'){getFrame(+r[1]).voxels.push({i:+r[2],p:r.slice(3,6).map(Number),damage:+r[6],lost:+r[7]})}}
let camera={center:[0,0,0],scale:1};
function fitCamera(points,w,h,padding=.84){if(!points.length){camera={center:[0,0,0],scale:1};return}const lo=[0,1,2].map(i=>Math.min(...points.map(p=>p[i]))),hi=[0,1,2].map(i=>Math.max(...points.map(p=>p[i])));camera.center=lo.map((v,i)=>(v+hi[i])/2);const e=hi.map((v,i)=>Math.max((v-lo[i])/2,.005)),ex=(e[0]+e[1])*.70710678,ey=(e[0]+e[1])*.40824829+e[2]*.81649658;camera.scale=padding*Math.min(w/(2*ex),h/(2*ey))}
function project(p,w,h,s){const a=p.map((v,i)=>v-camera.center[i]);return[w*.5+(a[0]+a[1])*.70710678*camera.scale,h*.57+((a[0]-a[1])*.40824829-a[2]*.81649658)*camera.scale]}
function boundsCorners(){if(!live.bounds)return[];const{min,max}=live.bounds;return[min[0],max[0]].flatMap(x=>[min[1],max[1]].flatMap(y=>[min[2],max[2]].map(z=>[x,y,z])))}
function drawWorldBounds(w,h){const pts=boundsCorners().map(p=>project(p,w,h,camera.scale)),edges=[[0,2],[0,4],[0,1],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];ctx.strokeStyle='#7790aa';ctx.lineWidth=1.3;ctx.globalAlpha=.9;for(const[a,b]of edges){ctx.beginPath();ctx.moveTo(...pts[a]);ctx.lineTo(...pts[b]);ctx.stroke()}ctx.globalAlpha=1;const tags=[['X−',0],['X+',4],['Y−',0],['Y+',2],['Z−',0],['Z+',1]];ctx.fillStyle='#b6c9dc';ctx.font='11px system-ui';for(const[tag,i]of tags)ctx.fillText(tag,pts[i][0]+4,pts[i][1]-4)}
function rotate(q,p){const[x,y,z,w]=q,[a,b,c]=p,tx=2*(y*c-z*b),ty=2*(z*a-x*c),tz=2*(x*b-y*a);return[a+w*tx+y*tz-z*ty,b+w*ty+z*tx-x*tz,c+w*tz+x*ty-y*tx]}
function heatColor(temp){const t=Math.max(0,Math.min(1,(temp-300)/150)),stops=[[38,112,218],[250,202,71],[225,54,48]],x=t*2,i=Math.min(1,Math.floor(x)),a=x-i;return `rgb(${stops[i].map((v,c)=>Math.round(v*(1-a)+stops[i+1][c]*a)).join(',')})`}
function drawCube(p,side,color,w,h,s){const d=side/2,pts=[[-d,-d,-d],[d,-d,-d],[d,d,-d],[-d,d,-d],[-d,-d,d],[d,-d,d],[d,d,d],[-d,d,d]].map(v=>project(p.map((x,i)=>x+v[i]),w,h,s));const faces=[[[0,1,5,4],.72],[[1,2,6,5],.9],[[4,5,6,7],1.12]];for(const [ids,shade] of faces){ctx.beginPath();ids.forEach((id,i)=>i?ctx.lineTo(...pts[id]):ctx.moveTo(...pts[id]));ctx.closePath();ctx.fillStyle=tint(color,shade);ctx.fill();ctx.strokeStyle='#263c53';ctx.lineWidth=.65;ctx.stroke()}}
function tint(color,k){let a;if(/^#[\da-f]{3}$/i.test(color)){a=color.slice(1).split('').map(v=>parseInt(v+v,16))}else if(/^#[\da-f]{6}$/i.test(color)){a=[1,3,5].map(i=>parseInt(color.slice(i,i+2),16))}else{a=color.match(/\d+/g)?.map(Number)||[150,150,150]}return `rgb(${a.map(v=>Math.max(0,Math.min(255,Math.round(v*k)))).join(',')})`}
function paintThermal(f,w,h){
 const cells=(f?.voxels||[]).slice().sort((a,b)=>a.i-b.i),side=.01,bp=boundsCorners();
 fitCamera(live.bounds?bp:cells.map(v=>v.p),w,h,.8);
 if(live.bounds)drawWorldBounds(w,h);
 if(cells.length){const min=Math.min(...cells.map(v=>v.temp)),max=Math.max(...cells.map(v=>v.temp));
  if(!live.bounds){const xs=cells.map(v=>v.p[0]),ys=cells.map(v=>v.p[1]),z0=Math.min(...cells.map(v=>v.p[2]))-side/2;const plane=[[Math.min(...xs)-.025,Math.min(...ys)-.025,z0],[Math.max(...xs)+.025,Math.min(...ys)-.025,z0],[Math.max(...xs)+.025,Math.max(...ys)+.025,z0],[Math.min(...xs)-.025,Math.max(...ys)+.025,z0]].map(p=>project(p,w,h,camera.scale));ctx.beginPath();plane.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fillStyle='#182638';ctx.fill();ctx.strokeStyle='#52657a';ctx.stroke()}
  const byId=new Map(cells.map(v=>[v.i,v]));ctx.lineWidth=3;ctx.strokeStyle='#a7bbcf';ctx.globalAlpha=.78;for(const [i,j] of (f?.bonds||[])){const a=byId.get(i),b=byId.get(j);if(!a||!b)continue;ctx.beginPath();ctx.moveTo(...project(a.p,w,h,camera.scale));ctx.lineTo(...project(b.p,w,h,camera.scale));ctx.stroke()}ctx.globalAlpha=1;for(const v of cells)drawCube(v.p,side*.92,heatColor(v.temp),w,h,camera.scale);
  ctx.fillStyle='#e8eef7';ctx.font='14px system-ui';ctx.fillText(`t = ${fmt(f.n*live.dt,3)} s · кадр ${f.n}/${live.steps}`,16,22);
 }
 if(!cells.length&&!live.bounds){ctx.fillStyle='#9aabc0';ctx.fillText('Ожидание теплового состояния…',20,32)}
 const ledger=f?.ledger,faceText=Object.entries(live.events).map(([face,n])=>`${face}: ${n}`).join(' · '),summary=ledger?`граница: удалено ${ledger.removed}, осталось ${ledger.remaining}, масса ${fmt(ledger.mass,5)} кг, U ${fmt(ledger.internal,3)} Дж, K ${fmt(ledger.kinetic,3)} Дж, p=(${ledger.momentum.map(v=>fmt(v,3)).join(', ')}) кг·м/с`:'';
 $('#live-caption').textContent=live.bounds?`Границы мира ${fmt(live.bounds.min[0],2)}…${fmt(live.bounds.max[0],2)} м · ${summary} · ${faceText}`:(cells.length?`3D кадр · ${Math.min(...cells.map(v=>v.temp)).toFixed(2)}–${Math.max(...cells.map(v=>v.temp)).toFixed(2)} K · связей: ${(f?.bonds||[]).length}`:'Ожидание теплового состояния…')
}
function paintLive(){
 const w=canvas.clientWidth,h=canvas.clientHeight,dpr=devicePixelRatio||1;if(!w||!h)return;
 if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#07101c';ctx.fillRect(0,0,w,h);const f=live.latest;
 if(live.mode==='thermal'){paintThermal(f,w,h);return}
 if(!f||!f.tool.length){ctx.fillStyle='#9aabc0';ctx.fillText('Ожидание первого сохранённого состояния…',20,32);return}
 const items=[];for(const v of f.voxels){const color=v.i>=4320?'#38bdf8':v.lost>=2?'#ef4444':v.lost?'#fb923c':v.damage>0?'#fbbf24':'#d17a38';items.push({p:v.p,c:color,z:v.p[0]-v.p[1]+v.p[2],side:v.i>=4320?.01:.008})}
 const q=f.tool.slice(3,7),pos=f.tool.slice(0,3);if(q.length===4)for(const cell of live.cells){const r=rotate(q,cell),p=pos.map((v,i)=>v+r[i]);items.push({p,c:'#e2e8f0',z:p[0]-p[1]+p[2],side:.008})}
 fitCamera(items.map(v=>v.p),w,h,.72);
 if(live.floor.length){const xs=live.floor.map(p=>p[0]),ys=live.floor.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),poly=[[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]].map(p=>project(p,w,h,camera.scale));ctx.beginPath();poly.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fillStyle='#202f40';ctx.fill();ctx.strokeStyle='#71869c';ctx.stroke()}
 items.sort((a,b)=>a.z-b.z);for(const v of items)drawCube(v.p,v.side,v.c,w,h,camera.scale);
 ctx.fillStyle='#e8eef7';ctx.font='14px system-ui';ctx.fillText(`t = ${fmt(f.n*live.dt,3)} s · кадр ${f.n}/${live.steps}`,16,22);$('#live-caption').textContent=`3D кадр решателя · разорвано связей: ${fmt(f.tool[9],0)} · скорость инструмента: ${fmt(f.tool[7])} м/с`
}
async function pollLive(){try{const suffix=live.experiment?'?experiment='+encodeURIComponent(live.experiment):'',s=await fetch('/api/status'+suffix,{cache:'no-store'}).then(r=>r.json());if(s.bytes<live.offset)liveReset();const qs=new URLSearchParams({offset:live.offset,limit:1048576,experiment:live.experiment}),chunk=await fetch('/capture?'+qs,{cache:'no-store'}).then(r=>r.text());live.offset+=new TextEncoder().encode(chunk).length;const parts=(live.carry+chunk).split('\n');live.carry=parts.pop()||'';parts.forEach(parseLine);paintLive();if(s.gif&&s.gif_mtime!==live.gifMtime){live.gifMtime=s.gif_mtime;const u='/gif'+(suffix||'?')+'&v='+encodeURIComponent(s.gif_mtime);animation.src=u;$('#gif-link').href=u}}catch(_){$('#live-caption').textContent='Нет соединения с CSV-потоком'}}
async function init(){
 const catalog=await fetch('/api/experiments',{cache:'no-store'}).then(r=>r.json());if(catalog.length){const select=$('#experiment');select.hidden=catalog.length<2;if(!select.options.length)catalog.forEach(e=>{const o=document.createElement('option');o.value=e.id;o.textContent=e.label;select.append(o)});if(!live.experiment||!catalog.some(e=>e.id===live.experiment))live.experiment=catalog[0].id;select.value=live.experiment;select.onchange=()=>{live.experiment=select.value;history.replaceState(null,'','?experiment='+encodeURIComponent(live.experiment));liveReset();init()}}
 const suffix=live.experiment?'?experiment='+encodeURIComponent(live.experiment):'',s=await fetch('/api/status'+suffix,{cache:'no-store'}).then(r=>r.json());
 setText('title',s.label);setText('source',`${s.file} · CSV: ${Number(s.bytes).toLocaleString('ru-RU')} байт`);
 $('#badge').textContent=s.gif?'GIF готова':'GIF не найдена';$('#badge').style.color=s.gif?'#8de0b0':'#f2c17e';
 $('#description').textContent=s.mode==='thermal'?'Живой тепловой профиль рассчитанных состояний REV.':'Зацикленная визуализация последней сохранённой записи физического расчёта.';
 if(s.gif){const url='/gif'+(suffix||'?')+'&v='+encodeURIComponent(s.gif_mtime||s.gif_bytes||Date.now());animation.src=url;$('#gif-link').href=url}
 else{$('#gif-error').style.display='block';animation.hidden=true}
 if(s.video){for(const id of ['recording-main','recording-aside'])$('#'+id).style.display='block';for(const a of document.querySelectorAll('#recording-main a,#recording-aside a'))a.href='/video'+suffix}
 if(s.mode==='csv'||s.mode==='thermal'){$('#live-preview').hidden=false;live.mode=s.mode==='thermal'?'thermal':'mechanical';window.addEventListener('resize',paintLive);await pollLive();if(!window.pollTimer)window.pollTimer=setInterval(pollLive,500)}
}
init().catch(()=>{$('#badge').textContent='Ошибка подключения';setText('source','Не удалось загрузить состояние просмотрщика')});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    source_path = DEFAULT_VIDEO
    mode = "video"
    label = "Сохранённая демонстрация контакта ножа"
    experiments = {}
    default_experiment = ""

    @classmethod
    def selected(cls, route):
        experiment_id = parse_qs(route.query).get("experiment", [cls.default_experiment])[0]
        return cls.experiments.get(experiment_id)

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
        if route.path == "/api/experiments":
            payload = json.dumps([
                {"id": item["id"], "label": item["label"], "mode": item["mode"]}
                for item in self.experiments.values()
            ], ensure_ascii=False).encode("utf-8")
            self.send_bytes(200, "application/json; charset=utf-8", payload)
            return
        selected = self.selected(route) if self.experiments else None
        source_path = selected["csv"] if selected else self.source_path
        mode = selected["mode"] if selected else self.mode
        label = selected["label"] if selected else self.label
        video_path = selected["video"] if selected else self.video_path()
        gif_path = selected["gif"] if selected else self.gif_path()
        if route.path == "/api/status":
            try:
                size = source_path.stat().st_size
            except FileNotFoundError:
                size = 0
            payload = json.dumps({
                "mode": mode,
                "label": label,
                "file": source_path.name,
                "bytes": size,
                "video": video_path.is_file(),
                "gif": gif_path.is_file(),
                "gif_bytes": gif_path.stat().st_size if gif_path.is_file() else 0,
                "gif_mtime": gif_path.stat().st_mtime_ns if gif_path.is_file() else 0,
            }, ensure_ascii=False).encode("utf-8")
            self.send_bytes(200, "application/json; charset=utf-8", payload)
            return
        if route.path == "/capture" and mode in ("csv", "thermal"):
            try:
                query = parse_qs(route.query)
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = min(4 * 1024 * 1024, max(1024, int(query.get("limit", ["1048576"])[0])))
                with source_path.open("rb") as source:
                    source.seek(min(offset, source_path.stat().st_size))
                    payload = source.read(limit)
            except FileNotFoundError:
                payload = b""
            except (OSError, ValueError):
                self.send_error(400, "Invalid capture offset")
                return
            self.send_bytes(200, "text/plain; charset=utf-8", payload)
            return
        if route.path == "/gif":
            try:
                payload = gif_path.read_bytes()
                self.send_bytes(200, "image/gif", payload)
            except OSError:
                self.send_error(404, "GIF animation not found")
            return
        if route.path == "/video":
            try:
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
    source.add_argument("--csv", type=Path, help="Append-only mechanical solver capture CSV")
    source.add_argument("--thermal-csv", type=Path, help="Append-only thermal transport capture CSV")
    parser.add_argument("--label", default="Демонстрация физического расчёта")
    parser.add_argument("--experiments", type=Path, help="JSON catalog of experiments, each with CSV/GIF/MP4 paths")
    args = parser.parse_args()
    Handler.mode = "thermal" if args.thermal_csv else "csv" if args.csv else "video"
    Handler.source_path = (args.thermal_csv or args.csv or args.video).resolve()
    Handler.label = args.label
    if args.experiments:
        catalog_path = args.experiments.resolve()
        try:
            raw = json.loads(catalog_path.read_text(encoding="utf-8"))
            entries = raw["experiments"]
            default_id = raw.get("default")
            if not entries:
                raise ValueError("catalog must contain at least one experiment")
            catalog = {}
            for entry in entries:
                experiment_id = entry["id"]
                mode = entry["mode"]
                if experiment_id in catalog or mode not in ("csv", "thermal"):
                    raise ValueError(f"duplicate id or unsupported mode in experiment {experiment_id!r}")
                base = catalog_path.parent
                csv_path = (base / entry["csv"]).resolve()
                gif_path = (base / entry.get("gif", str(Path(entry["csv"]).with_suffix(".gif")))).resolve()
                video_path = (base / entry.get("video", str(Path(entry["csv"]).with_suffix(".mp4")))).resolve()
                if not csv_path.parent.is_dir():
                    raise ValueError(f"CSV directory does not exist for {experiment_id}: {csv_path.parent}")
                catalog[experiment_id] = {"id": experiment_id, "label": entry["label"], "mode": mode,
                                          "csv": csv_path, "gif": gif_path, "video": video_path}
            Handler.experiments = catalog
            Handler.default_experiment = default_id if default_id in catalog else next(iter(catalog))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            parser.error(f"invalid experiment catalog: {error}")
    if Handler.mode == "video" and not Handler.source_path.is_file():
        parser.error(f"video file does not exist: {Handler.source_path}")
    if Handler.mode in ("csv", "thermal") and not Handler.source_path.parent.is_dir():
        parser.error(f"CSV parent directory does not exist: {Handler.source_path.parent}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Viewer listening on http://127.0.0.1:{args.port} ({Handler.mode})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
