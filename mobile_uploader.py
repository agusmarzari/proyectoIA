#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subidor móvil local (Flask) para clasificar fotos desde el celular.
- Página móvil con dos formularios: Cámara y Galería (compatible iPhone)
- Soporta HEIC (iPhone) y JPG/PNG
- Usa el MISMO extractor de features que el entrenamiento (features_vision.py)
- Guarda imagen y JSON en 'inbox/' y clasifica con models/kmeans_piezas.joblib
Requisitos:
  pip install flask qrcode[pil] opencv-python scikit-image scikit-learn joblib numpy pillow pillow-heif
Ejecución:
  python mobile_uploader.py --host 0.0.0.0 --port 5006 --save_dir inbox
"""

import argparse, io, json, socket, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string, redirect, url_for
from werkzeug.utils import secure_filename
from joblib import load
import qrcode
import cv2

from features_vision import extract_features_image_training as extract_features_image  # 👈

MODELS_DIR = Path("models")
KMEANS_PATH = MODELS_DIR / "kmeans_piezas.joblib"
CLASSES_ORDER = ["tornillos","clavos","arandelas","tuercas"]

def get_local_ips():
    ips = set()
    try: ips.add(socket.gethostbyname(socket.gethostname()))
    except: pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0]); s.close()
    except: pass
    ips = [ip for ip in ips if ip and not ip.startswith("127.")]
    return ips or ["127.0.0.1"]

def load_kmeans_bundle(path: Path):
    if not path.exists():
        raise SystemExit(f"No se encontró el modelo: {path}")
    return load(path)

def predict_label(bundle, feat):
    kmeans = bundle["kmeans"]; scaler = bundle["scaler"]
    pca = bundle.get("pca", None)
    cluster_to_class = bundle["cluster_to_class"]; class_names = bundle["classes"]

    fs = scaler.transform(feat.reshape(1,-1))
    if pca is not None:
        fs = pca.transform(fs)

    clus = int(kmeans.predict(fs)[0])
    lab = cluster_to_class.get(clus, None)
    if isinstance(lab, int):
        lab = class_names[lab]
    lab = lab or f"cluster_{clus}"
    for std in CLASSES_ORDER:
        if std.lower() in str(lab).lower() or str(lab).lower() in std.lower():
            return std, clus
    return str(lab), clus

HTML = """
<!doctype html><html lang="es"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Subir foto — Agente IA</title>
<style>
  :root { --bg:#0f172a; --card:#111827; --line:#1f2937; --txt:#e5e7eb; --muted:#9ca3af; --btn:#3b82f6; --ok:#10b981; }
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto; margin:0; background:var(--bg); color:var(--txt)}
  .wrap{max-width:680px; margin:0 auto; padding:24px}
  h1{font-size:20px; margin:0 0 8px}
  .card{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; margin:16px 0}
  .btn{display:inline-block; background:var(--btn); color:#fff; padding:12px 16px; border-radius:12px; text-decoration:none; border:0; font-weight:600}
  .btn:active{transform:scale(.98)}
  input[type=file]{width:100%; padding:12px; border-radius:12px; border:1px dashed var(--line); background:#0b1220; color:var(--txt)}
  img{max-width:100%; border-radius:12px; border:1px solid var(--line)}
  .ok{color:var(--ok); font-weight:700}
  .meta{font-size:12px; color:var(--muted)}
  .row{display:flex; gap:12px; flex-wrap:wrap}
</style>
<script>
  function resetAfterSubmit(form){ setTimeout(()=>{ try{ form.reset(); }catch(e){} }, 400); }
</script>
</head><body>
<div class="wrap">
  <h1>📸 Subí una foto para clasificar</h1>

  <div class="card">
    <p class="meta">Elegí <b>Cámara</b> o <b>Galería</b>. En iPhone, si la cámara no vuelve a abrir tras la primera foto, cerrá la pestaña y reingresá, o usá Galería.</p>
    <div class="row">
      <form action="/upload" method="post" enctype="multipart/form-data" onsubmit="resetAfterSubmit(this)">
        <input name="file" type="file" accept="image/*" required>
        <p style="margin-top:8px"><button class="btn" type="submit">Usar cámara</button></p>
      </form>
      <form action="/upload" method="post" enctype="multipart/form-data" onsubmit="resetAfterSubmit(this)">
        <input name="file" type="file" accept="image/*" required>
        <p style="margin-top:8px"><button class="btn" type="submit">Subir desde galería</button></p>
      </form>
    </div>
  </div>

  {% if result %}
  <div class="card">
    <p class="ok">✅ Predicción: {{ result.label }}</p>
    <p class="meta">Archivo: {{ result.filename }} · Cluster: {{ result.cluster }}</p>
    {% if result.preview %}<img src="/last" alt="preview"/>{% endif %}
  </div>
  {% endif %}
</div>
</body></html>
"""

app = Flask(__name__)
BUNDLE = load_kmeans_bundle(KMEANS_PATH)
LAST_PREVIEW = None

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML, result=None)

@app.route("/upload", methods=["POST"])
def upload_page():
    try:
        f = request.files.get("file")
        if not f:
            return render_template_string(HTML, result={"label":"No se recibió archivo","filename":"-","cluster":"-","preview":False})
        save_dir = Path(app.config["SAVE_DIR"]); save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = secure_filename(f.filename) or "foto.jpg"
        out = save_dir / f"{ts}_{name}"
        data = f.read()
        out.write_bytes(data)

        feat, img = extract_features_image(data)
        if feat is None:
            msg = "Formato no soportado o imagen vacía (usá JPG/PNG o ‘Más compatible’ en iPhone)."
            return render_template_string(HTML, result={"label": msg, "filename": out.name, "cluster":"-", "preview": False})

        label, clus = predict_label(BUNDLE, feat)
        # preview
        preview = save_dir / "last_preview.jpg"
        if img is not None:
            cv2.imwrite(str(preview), img)
            global LAST_PREVIEW; LAST_PREVIEW = preview

        meta = {"filename": out.name, "label": label, "cluster": clus, "preview": True}
        (save_dir / f"{out.stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return render_template_string(HTML, result=meta)

    except Exception as e:
        traceback.print_exc()
        msg = f"Error interno: {type(e).__name__}"
        return render_template_string(HTML, result={"label": msg, "filename":"-", "cluster":"-", "preview": False}), 500

@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        f = request.files.get("file")
        if not f:
            return jsonify(error="no_file"), 400
        save_dir = Path(app.config["SAVE_DIR"]); save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = secure_filename(f.filename) or "foto.jpg"
        out = save_dir / f"{ts}_{name}"
        data = f.read()
        out.write_bytes(data)

        feat, _ = extract_features_image(data)
        if feat is None:
            return jsonify(error="unsupported_format"), 415

        label, clus = predict_label(BUNDLE, feat)
        meta = {"filename": out.name, "label": label, "cluster": clus}
        (save_dir / f"{out.stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return jsonify(meta)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error="internal_error"), 500

@app.route("/last", methods=["GET"])
def last_image():
    if LAST_PREVIEW and Path(LAST_PREVIEW).exists():
        return send_file(str(LAST_PREVIEW), mimetype="image/jpeg")
    return ("", 204)

@app.route("/qr.png", methods=["GET"])
def qr_png():
    url = app.config["APP_URL"]
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--save_dir", default="inbox")
    args = parser.parse_args()

    app.config["SAVE_DIR"] = args.save_dir

    ips = get_local_ips()
    url = f"http://{ips[0]}:{args.port}/"
    app.config["APP_URL"] = url

    print("\n📡 Subidor móvil listo")
    print("   Abrí desde el CELU (misma red o hotspot):")
    for ip in ips:
        print(f"   → http://{ip}:{args.port}/")
    print(f"   QR: http://{ips[0]}:{args.port}/qr.png\n")

    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
