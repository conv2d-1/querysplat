#!/usr/bin/env python3
"""
Interactive manual person-mask editor (FastAPI + HTML canvas).

Avoids Gradio ImageEditor bugs on gradio 4.44.x (HTTP 500 on page load).

Usage:
    python hAlgorithm/script/infer/motion_head/manual_person_mask_editor.py \\
        --clip_dirs results/.../clip0023 results/.../clip0025 --port 7860

Then SSH tunnel from local machine:
    ssh -L 7860:127.0.0.1:7860 -p 30390 tcchen@172.20.11.68

Open in local browser:
    http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

MANUAL_MASK_SUFFIX = "_manual_ref_mask.png"
YOLO_MASK_SUFFIX = "_yolo_ref_mask.png"

HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Manual Person Mask Editor</title>
  <style>
    body { font-family: sans-serif; margin: 16px; background: #111; color: #eee; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
    .panel { background: #1b1b1b; padding: 12px; border-radius: 8px; }
    canvas { border: 1px solid #444; cursor: crosshair; max-width: 100%; background: #000; }
    label { display: inline-block; margin-right: 12px; }
    button { margin-right: 8px; margin-top: 8px; padding: 6px 12px; }
    #info, #status { white-space: pre-wrap; font-family: monospace; font-size: 12px; }
    #status { color: #9fd; min-height: 48px; }
  </style>
</head>
<body>
  <h2>Manual Person Mask Editor</h2>
  <p>在 RGB 上涂抹 <b style="color:#f55">红色</b> 区域作为 person mask。滚轮或滑条调画笔大小。保存后写入 <code>mask_debug/*_manual_ref_mask.png</code></p>
  <div class="panel">
    <label>Clip <select id="clipSelect"></select></label>
    <label>Brush <input id="brushSize" type="range" min="4" max="120" value="24"></label>
    <span id="brushLabel">24px</span>
    <label><input id="eraseMode" type="checkbox"> Eraser</label>
    <br>
    <button id="btnYolo">Reset from YOLO</button>
    <button id="btnClear">Clear</button>
    <button id="btnSave">Save manual mask</button>
  </div>
  <div class="row">
    <div class="panel">
      <div>Canvas (RGB + mask overlay)</div>
      <canvas id="view"></canvas>
    </div>
    <div class="panel" style="min-width:320px">
      <div>Info</div>
      <div id="info"></div>
      <div style="margin-top:12px">Status</div>
      <div id="status"></div>
    </div>
  </div>
<script>
let clips = [];
let current = null;
let rgbImg = new Image();
let mask = null; // Uint8Array H*W, 0/255
let drawing = false;
let hw = [0, 0];

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const clipSelect = document.getElementById('clipSelect');
const brushSize = document.getElementById('brushSize');
const brushLabel = document.getElementById('brushLabel');
const eraseMode = document.getElementById('eraseMode');
const info = document.getElementById('info');
const status = document.getElementById('status');

function setStatus(msg) { status.textContent = msg; }

function redraw() {
  if (!rgbImg.complete || !mask) return;
  const [h, w] = hw;
  canvas.width = w; canvas.height = h;
  ctx.drawImage(rgbImg, 0, 0, w, h);
  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 127) {
      const p = i * 4;
      data[p]   = Math.min(255, data[p]   * 0.45 + 255 * 0.55);
      data[p+1] = Math.min(255, data[p+1] * 0.45 +  50 * 0.55);
      data[p+2] = Math.min(255, data[p+2] * 0.45 +  50 * 0.55);
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

function canvasPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return [
    Math.max(0, Math.min(canvas.width - 1, Math.round((evt.clientX - rect.left) * sx))),
    Math.max(0, Math.min(canvas.height - 1, Math.round((evt.clientY - rect.top) * sy))),
  ];
}

function paint(x, y) {
  const r = parseInt(brushSize.value, 10);
  const erase = eraseMode.checked;
  const val = erase ? 0 : 255;
  const w = canvas.width, h = canvas.height;
  const r2 = r * r;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      if (dx*dx + dy*dy > r2) continue;
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      mask[yy * w + xx] = val;
    }
  }
  redraw();
}

canvas.addEventListener('mousedown', (e) => { drawing = true; const [x,y]=canvasPos(e); paint(x,y); });
canvas.addEventListener('mousemove', (e) => { if (!drawing) return; const [x,y]=canvasPos(e); paint(x,y); });
window.addEventListener('mouseup', () => { drawing = false; });

brushSize.addEventListener('input', () => { brushLabel.textContent = brushSize.value + 'px'; });
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  let v = parseInt(brushSize.value, 10) + (e.deltaY > 0 ? -2 : 2);
  v = Math.max(4, Math.min(120, v));
  brushSize.value = v; brushLabel.textContent = v + 'px';
}, { passive: false });

function base64ToBytes(b64) {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

async function loadMaskBytes(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('Failed to load mask: ' + url);
  return new Uint8Array(await resp.arrayBuffer());
}

async function loadClip(name) {
  setStatus('Loading ' + name + ' ...');
  const meta = await fetch('/api/clip/' + encodeURIComponent(name)).then(r => r.json());
  current = meta;
  info.textContent = JSON.stringify(meta, null, 2);
  rgbImg = new Image();
  const maskPromise = loadMaskBytes('/api/clip/' + encodeURIComponent(name) + '/mask');
  rgbImg.onload = async () => {
    hw = [meta.height, meta.width];
    mask = await maskPromise;
    if (mask.length !== meta.width * meta.height) {
      throw new Error('Mask size mismatch: ' + mask.length + ' vs ' + (meta.width * meta.height));
    }
    redraw();
    setStatus('Loaded. fg_pixels=' + meta.fg_pixels);
  };
  rgbImg.onerror = () => setStatus('Failed to load RGB image.');
  rgbImg.src = 'data:image/jpeg;base64,' + meta.rgb_b64;
}

async function init() {
  clips = await fetch('/api/clips').then(r => r.json());
  clipSelect.innerHTML = '';
  clips.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.name; opt.textContent = c.name + ' (clip_idx=' + c.clip_idx + ')';
    clipSelect.appendChild(opt);
  });
  clipSelect.onchange = () => loadClip(clipSelect.value);
  await loadClip(clips[0].name);
}

document.getElementById('btnYolo').onclick = async () => {
  if (!current) return;
  try {
    mask = await loadMaskBytes('/api/clip/' + encodeURIComponent(current.name) + '/mask/yolo');
    redraw();
    const fg = mask.reduce((a, v) => a + (v > 127 ? 1 : 0), 0);
    setStatus('Reset from YOLO. fg_pixels=' + fg);
  } catch (err) {
    setStatus('Reset failed: ' + err);
  }
};

document.getElementById('btnClear').onclick = () => {
  if (!mask) return;
  mask.fill(0); redraw(); setStatus('Cleared mask.');
};

document.getElementById('btnSave').onclick = async () => {
  if (!current || !mask) return;
  setStatus('Saving...');
  try {
    const resp = await fetch('/api/save/' + encodeURIComponent(current.name), {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: mask,
    });
    const out = await resp.json();
    if (!resp.ok) throw new Error(out.detail || JSON.stringify(out));
    setStatus('Saved → ' + out.manual_path + '\nfg_pixels=' + out.fg_pixels);
  } catch (err) {
    setStatus('Save failed: ' + err);
  }
};

init().catch(err => setStatus('Init failed: ' + err));
</script>
</body>
</html>
"""


def parse_args():
    p = argparse.ArgumentParser(description="Manual person mask editor (FastAPI)")
    p.add_argument("--clip_dirs", nargs="+", required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_clip(clip_dir: str) -> dict:
    root = os.path.abspath(os.path.expanduser(clip_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Clip dir not found: {root}")
    seq_name = Path(root).name
    summary_path = os.path.join(root, "summary.json")
    meta_path = os.path.join(root, "source_meta.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing summary.json under {root}")

    summary = _read_json(summary_path)
    ref_rgb = summary.get("ref_rgb")
    if not ref_rgb or not os.path.isfile(ref_rgb):
        if os.path.isfile(meta_path):
            rgb_paths = _read_json(meta_path).get("rgb_paths") or []
            if rgb_paths:
                ref_rgb = rgb_paths[0]
    if not ref_rgb or not os.path.isfile(ref_rgb):
        frame_guess = os.path.join(root, "_frames", "frame_000001.png")
        if os.path.isfile(frame_guess):
            ref_rgb = frame_guess
    if not ref_rgb or not os.path.isfile(ref_rgb):
        raise FileNotFoundError(f"Cannot resolve ref RGB for {root}")

    mask_dbg = os.path.join(root, "mask_debug")
    return dict(
        clip_dir=root,
        seq_name=seq_name,
        clip_idx=summary.get("clip_idx"),
        ref_rgb=ref_rgb,
        manual_path=os.path.join(mask_dbg, f"{seq_name}{MANUAL_MASK_SUFFIX}"),
        yolo_path=os.path.join(mask_dbg, f"{seq_name}{YOLO_MASK_SUFFIX}"),
        summary=summary,
    )


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f"Failed to read RGB: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask_gray(path: str | None, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if path and os.path.isfile(path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask
    return np.zeros((h, w), dtype=np.uint8)


def encode_jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_mask_b64(mask: np.ndarray) -> str:
    return base64.b64encode(mask.astype(np.uint8).tobytes()).decode("ascii")


def decode_mask_b64(data: str, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    raw = base64.b64decode(data.encode("ascii"))
    mask = np.frombuffer(raw, dtype=np.uint8)
    if mask.size != h * w:
        raise ValueError(f"Mask size mismatch: got {mask.size}, expected {h*w}")
    return mask.reshape(h, w)


def clip_payload(clip: dict, *, prefer_manual: bool) -> dict:
    rgb = load_rgb(clip["ref_rgb"])
    h, w = rgb.shape[:2]
    init_path = clip["manual_path"] if prefer_manual and os.path.isfile(clip["manual_path"]) else clip["yolo_path"]
    if prefer_manual and os.path.isfile(clip["manual_path"]):
        src = "manual"
    elif clip["yolo_path"] and os.path.isfile(clip["yolo_path"]):
        src = "yolo"
    else:
        src = "empty"
    mask = load_mask_gray(init_path if src != "empty" else None, (h, w))
    return dict(
        name=clip["seq_name"],
        clip_idx=clip.get("clip_idx"),
        clip_dir=clip["clip_dir"],
        ref_rgb=clip["ref_rgb"],
        save_to=clip["manual_path"],
        init_mask=src,
        width=w,
        height=h,
        fg_pixels=int((mask > 127).sum()),
        rgb_b64=encode_jpeg_b64(rgb),
    )


def _write_manual_mask(clip: dict, mask: np.ndarray) -> dict:
    if not (mask > 127).any():
        raise HTTPException(400, "Mask is empty.")
    rgb = load_rgb(clip["ref_rgb"])
    h, w = rgb.shape[:2]
    if mask.shape[:2] != (h, w):
        raise HTTPException(400, f"Mask shape {mask.shape[:2]} != RGB {(h, w)}")
    os.makedirs(os.path.dirname(clip["manual_path"]), exist_ok=True)
    cv2.imwrite(clip["manual_path"], mask)
    vis_path = os.path.join(
        os.path.dirname(clip["manual_path"]),
        f"{clip['seq_name']}_manual_ref_mask_vis.png",
    )
    vis = np.zeros(mask.shape[:2], dtype=np.uint8)
    vis[mask > 127] = 255
    cv2.imwrite(vis_path, vis)
    overlay = rgb.copy().astype(np.float32)
    fg = mask > 127
    overlay[fg] = overlay[fg] * 0.45 + np.array([255, 50, 50], np.float32) * 0.55
    overlay_path = os.path.join(
        os.path.dirname(clip["manual_path"]),
        f"{clip['seq_name']}_manual_overlay.jpg",
    )
    cv2.imwrite(overlay_path, cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return dict(
        manual_path=clip["manual_path"],
        overlay_path=overlay_path,
        fg_pixels=int(fg.sum()),
    )


def create_app(clips: list[dict]) -> FastAPI:
    app = FastAPI(title="Manual Person Mask Editor")
    by_name = {c["seq_name"]: c for c in clips}

    class SaveBody(BaseModel):
        name: str
        mask_b64: str

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/api/clips")
    def list_clips():
        return [{"name": c["seq_name"], "clip_idx": c.get("clip_idx")} for c in clips]

    @app.get("/api/clip/{name}")
    def get_clip(name: str):
        clip = by_name.get(name)
        if clip is None:
            raise HTTPException(404, f"Unknown clip: {name}")
        return clip_payload(clip, prefer_manual=True)

    @app.get("/api/clip/{name}/mask")
    def get_clip_mask(name: str):
        clip = by_name.get(name)
        if clip is None:
            raise HTTPException(404, f"Unknown clip: {name}")
        payload = clip_payload(clip, prefer_manual=True)
        rgb = load_rgb(clip["ref_rgb"])
        h, w = rgb.shape[:2]
        if payload["init_mask"] == "manual" and os.path.isfile(clip["manual_path"]):
            mask = load_mask_gray(clip["manual_path"], (h, w))
        elif clip["yolo_path"] and os.path.isfile(clip["yolo_path"]):
            mask = load_mask_gray(clip["yolo_path"], (h, w))
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        return Response(content=mask.astype(np.uint8).tobytes(), media_type="application/octet-stream")

    @app.get("/api/clip/{name}/mask/yolo")
    def get_clip_mask_yolo(name: str):
        clip = by_name.get(name)
        if clip is None:
            raise HTTPException(404, f"Unknown clip: {name}")
        rgb = load_rgb(clip["ref_rgb"])
        h, w = rgb.shape[:2]
        mask = load_mask_gray(clip["yolo_path"], (h, w))
        return Response(content=mask.astype(np.uint8).tobytes(), media_type="application/octet-stream")

    @app.get("/api/clip/{name}/yolo")
    def get_clip_yolo(name: str):
        clip = by_name.get(name)
        if clip is None:
            raise HTTPException(404, f"Unknown clip: {name}")
        rgb = load_rgb(clip["ref_rgb"])
        h, w = rgb.shape[:2]
        mask = load_mask_gray(clip["yolo_path"], (h, w))
        return dict(fg_pixels=int((mask > 127).sum()))

    @app.post("/api/save/{name}")
    async def save_mask_raw(name: str, request: Request):
        clip = by_name.get(name)
        if clip is None:
            raise HTTPException(404, f"Unknown clip: {name}")
        rgb = load_rgb(clip["ref_rgb"])
        h, w = rgb.shape[:2]
        raw = await request.body()
        expected = h * w
        if len(raw) != expected:
            raise HTTPException(400, f"Mask bytes {len(raw)} != expected {expected}")
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
        return _write_manual_mask(clip, mask)

    @app.post("/api/save")
    def save_mask_b64(body: SaveBody):
        clip = by_name.get(body.name)
        if clip is None:
            raise HTTPException(404, f"Unknown clip: {body.name}")
        rgb = load_rgb(clip["ref_rgb"])
        h, w = rgb.shape[:2]
        mask = decode_mask_b64(body.mask_b64, (h, w))
        return _write_manual_mask(clip, mask)

    return app


def main():
    args = parse_args()
    clips = [discover_clip(d) for d in args.clip_dirs]
    print("Manual mask editor (FastAPI)")
    print(f"Listen: http://{args.host}:{args.port}")
    print("SSH example:")
    print(f"  ssh -L {args.port}:127.0.0.1:{args.port} -p 30390 tcchen@172.20.11.68")
    print("Then open locally: http://127.0.0.1:7860")
    for c in clips:
        print(f"  - {c['seq_name']} (clip_idx={c.get('clip_idx')})")
    app = create_app(clips)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
