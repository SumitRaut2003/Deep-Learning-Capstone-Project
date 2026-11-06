"""Helmet Compliance Monitor - web backend.

Run:  python app.py            (then open http://127.0.0.1:5000)
Env:  HELMET_MODEL=path/to/best.onnx   HELMET_IMGSZ=640
"""
import argparse, csv, os, threading, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get('HELMET_MODEL', BASE / 'best.onnx'))
IMGSZ = int(os.environ.get('HELMET_IMGSZ', 640))
SNAP_DIR = BASE / 'snapshots'
LOG_CSV = BASE / 'violations.csv'
SNAP_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    raise SystemExit(f'Model not found: {MODEL_PATH}\nCopy best.onnx into this folder (next to app.py).')

app = Flask(__name__)
model = YOLO(str(MODEL_PATH), task='detect')
infer_lock = threading.Lock()
log_lock = threading.Lock()

if not LOG_CSV.exists():
    LOG_CSV.write_text('timestamp,snapshot,no_helmet_count\n')


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/health')
def health():
    return jsonify(model=MODEL_PATH.name, imgsz=IMGSZ, classes=list(model.names.values()))


@app.post('/api/detect')
def detect():
    ch = float(request.args.get('ch', 0.6))    # min confidence to ACCEPT "helmet"
    cn = float(request.args.get('cn', 0.3))    # min confidence to flag "no helmet"
    img = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error='could not decode image'), 400
    H, W = img.shape[:2]

    t0 = time.perf_counter()
    with infer_lock:
        r = model(img, conf=min(ch, cn), imgsz=IMGSZ, verbose=False)[0]
    model_ms = (time.perf_counter() - t0) * 1000

    helmets, nohelmets = [], []
    for b in r.boxes:
        box = b.xyxy[0].tolist()
        name = r.names[int(b.cls[0])]
        conf = float(b.conf[0])
        if name == 'helmet' and conf >= ch:
            helmets.append((box, conf))
        elif name != 'helmet' and conf >= cn:
            nohelmets.append((box, conf))

    # Safety-first: where both overlap on one head, trust "no helmet" unless helmet is far more confident
    kept = [(hb, hc) for hb, hc in helmets
            if not any(iou(hb, nb) > 0.4 and nc * 1.5 >= hc for nb, nc in nohelmets)]

    def pack(items, label):
        return [{'label': label, 'conf': round(c, 3),
                 'box': [round(bb[0] / W, 4), round(bb[1] / H, 4), round(bb[2] / W, 4), round(bb[3] / H, 4)]}
                for bb, c in items]

    dets = pack(kept, 'helmet') + pack(nohelmets, 'no_helmet')
    status = 'violation' if nohelmets else ('compliant' if kept else 'none')
    return jsonify(dets=dets, status=status, model_ms=round(model_ms, 1), width=W, height=H,
                   counts={'helmet': len(kept), 'no_helmet': len(nohelmets)})


@app.post('/api/violation')
def violation():
    data = request.get_data()
    n = int(request.args.get('n', 1))
    now = datetime.now()
    name = now.strftime('%Y%m%d_%H%M%S_') + f'{now.microsecond // 1000:03d}.jpg'
    (SNAP_DIR / name).write_bytes(data)
    with log_lock, open(LOG_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([now.isoformat(timespec='seconds'), name, n])
    return jsonify(ok=True, snapshot=name)


@app.get('/api/violations')
def violations():
    with log_lock, open(LOG_CSV, newline='') as f:
        rows = list(csv.DictReader(f))
    return jsonify([{'time': r['timestamp'], 'snapshot': r['snapshot'], 'count': int(r['no_helmet_count'])}
                    for r in rows[-20:]][::-1])


@app.get('/api/violations.csv')
def violations_csv():
    return send_file(LOG_CSV, mimetype='text/csv', as_attachment=True, download_name='violations.csv')


@app.get('/snapshots/<path:fn>')
def snapshot(fn):
    return send_from_directory(SNAP_DIR, fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5000)
    a = ap.parse_args()
    print(f'Model: {MODEL_PATH.name} | open http://{a.host}:{a.port}')
    app.run(host=a.host, port=a.port, threaded=True)
