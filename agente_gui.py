#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agente GUI – Visión (KMeans bundle) + Voz (KNN) + Bayes
Botones:
- Ejecutar: toma 10 imágenes aleatorias de dataset_mix, clasifica y guarda conteo.
- Escuchar: voz (proporcion / contar / salir) con fallback por teclado.
- Subir imagen: clasifica una imagen suelta usando el bundle.
Requiere:
  - kmeans_bundle.joblib  (generado por kmeans_mejorado.py)
  - cluster_to_class.json (idealmente mapea 0..3 -> 'tornillos','tuercas','arandelas','clavos')
  - bayes_estimator.py    (tus funciones posterior_over_boxes, expected_proportions)
  - voice_knn.py          (opcional; si falla, usa popup de texto)
"""

import os, glob, json, random, io
import numpy as np
import PySimpleGUI as sg
import cv2
from joblib import load
from skimage.feature import local_binary_pattern

# ---------------- Config ----------------
DATASET_DIR = "dataset_mix"
BUNDLE_PATH = "kmeans_bundle.joblib"
MAPJSON     = "cluster_to_class.json"
VOICE_MODEL = "knn_voice.joblib"   # el modelo que entrenaste con tu script original


# Orden canónico que espera Bayes:
CLASS_ORDER = ["tornillos", "clavos", "arandelas", "tuercas"]

# Tamaño / features (deben coincidir con el entrenamiento)
IMG_SIZE = (256, 256)
LBP_P, LBP_R, LBP_METHOD = 8, 1, "uniform"
HSV_BINS = 16
# ----------------------------------------


# --------------- Utilidades de imagen / features ---------------
def preprocess_mask(gray):
    """Fondo claro: máscara que preserva agujeros (tuerca/arandela)."""
    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = th.astype(np.uint8)
    # abrir suave (no cerrar para no tapar agujeros)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return th

    hier = hier[0]  # [Next, Prev, FirstChild, Parent]
    mask = np.zeros_like(th, dtype=np.uint8)
    for i, cnt in enumerate(cnts):
        parent = hier[i][3]
        if parent == -1:
            cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)  # externo
        else:
            cv2.drawContours(mask, [cnt], -1, 0, thickness=-1)  # agujero
    return mask


def hu_moments_from_mask(mask):
    m = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu.astype(np.float32)


def lbp_hist(gray):
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2  # uniform
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def hsv_hist(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hs = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [HSV_BINS], [0, 256]).flatten().astype(np.float32)
        hist /= (hist.sum() + 1e-12)
        hs.append(hist)
    return np.hstack(hs).astype(np.float32)


def extract_features_img(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No puedo leer {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = preprocess_mask(gray)
    hu  = hu_moments_from_mask(mask)
    lbp = lbp_hist(gray)
    hsv = hsv_hist(img)
    feat = np.hstack([hu, lbp, hsv]).astype(np.float32)
    return feat, mask, img
# ---------------------------------------------------------------


# --------------- Carga de bundle y mapping ---------------------
def load_bundle_and_mapping():
    if not os.path.exists(BUNDLE_PATH):
        sg.popup_error("No encuentro kmeans_bundle.joblib. Corré primero kmeans_mejorado.py")
        raise SystemExit
    bundle = load(BUNDLE_PATH)
    scaler, pca, kmeans = bundle["scaler"], bundle["pca"], bundle["kmeans"]

    clmap = {}
    if os.path.exists(MAPJSON):
        with open(MAPJSON, "r", encoding="utf-8") as f:
            clmap = json.load(f)
    return scaler, pca, kmeans, clmap


def resolve_label_name(lab, clmap):
    """Devuelve nombre de clase a partir del índice de cluster."""
    name = f"cluster_{lab}"
    # admitir claves "2" o 2
    if clmap:
        name = clmap.get(str(lab), clmap.get(lab, name))
    # normalizar a minúsculas sin tildes simples (por si el JSON ya tiene nombres reales)
    return str(name).strip().lower()
# ---------------------------------------------------------------


# --------------- Clasificación y muestreo ----------------------
def classify_image(path, scaler, pca, kmeans, clmap, save_debug=True):
    feat, mask, img = extract_features_img(path)
    xs = scaler.transform(feat.reshape(1, -1))
    xf = pca.transform(xs) if pca is not None else xs
    lab = int(kmeans.predict(xf)[0])
    name = resolve_label_name(lab, clmap)

    # guardar máscara debug
    if save_debug:
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(f"debug_mask_{base}.png", (mask * 255).astype("uint8"))
    return lab, name


def list_images_dataset():
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp","*.JPG","*.JPEG","*.PNG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(DATASET_DIR, e)))
    return files


def sample_and_classify_10(scaler, pca, kmeans, clmap):
    files = list_images_dataset()
    if len(files) < 10:
        raise RuntimeError(f"Necesito al menos 10 imágenes en {DATASET_DIR}")
    sample = random.sample(files, 10)
    counts = {}
    per_image = []
    for p in sample:
        _, name = classify_image(p, scaler, pca, kmeans, clmap, save_debug=False)
        per_image.append((p, name))
        counts[name] = counts.get(name, 0) + 1
    return counts, per_image
# ---------------------------------------------------------------


# --------------- Bayes y voz ----------------------------------
def run_bayes(counts_by_name):
    # transformar al orden canónico que espera Bayes
    from importlib import import_module
    be = import_module("bayes_estimator")
    vec = [counts_by_name.get(cls, 0) for cls in CLASS_ORDER]
    posterior = be.posterior_over_boxes(vec)
    exp_prop  = be.expected_proportions(posterior)
    # caja MAP:
    caja_idx = int(np.argmax(posterior))
    cajas = ["a","b","c","d"]
    caja = cajas[caja_idx] if caja_idx < len(cajas) else f"idx_{caja_idx}"
    return vec, posterior, exp_prop, caja

def crop_to_max_energy(y, sr, win_sec=1.2, hop_sec=0.05):
    import numpy as np
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if len(y) <= win: 
        return y
    # energía RMS por ventana deslizante
    best_s, best_e, best_rms = 0, win, -1.0
    for s in range(0, len(y)-win+1, hop):
        e = s + win
        rms = float(np.sqrt(np.mean(y[s:e]**2)))
        if rms > best_rms:
            best_rms, best_s, best_e = rms, s, e
    return y[best_s:best_e]


def listen_command(seconds=1.8, rate=16000):
    """
    Usa TU pipeline original de voz:
    - voice_knn.demo(modelo, seconds, rate)
    - Siempre devuelve la clase predicha por tu KNN entrenado.
    """
    import os
    import PySimpleGUI as sg
    try:
        import voice_knn  # <- tu archivo voice_knn.py
    except Exception as e:
        sg.popup_error(f"No pude importar voice_knn.py: {e}")
        return None

    if not os.path.exists(VOICE_MODEL):
        sg.popup_error(f"No encuentro el modelo de voz: {VOICE_MODEL}")
        return None

    # Ventanita de “Grabando…”
    sg.popup_quick_message(
        "🎙️ Grabando...",
        auto_close=True,
        auto_close_duration=seconds,
        no_titlebar=True,
        keep_on_top=True
    )

    try:
        pred = voice_knn.demo(VOICE_MODEL, seconds=seconds, rate=rate)
        # Normalizamos a minúsculas por las dudas
        return (pred or "").strip().lower()
    except Exception as e:
        sg.popup_error(f"Error al ejecutar demo() de voice_knn: {e}")
        return None




# --------------- GUI -------------------------------------------
def im_to_bytes(img_bgr, max_w=380, max_h=380):
    """Convierte BGR a PNG bytes para Image element (redimensiona manteniendo aspecto)."""
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale != 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _, buff = cv2.imencode(".png", img_rgb)
    return buff.tobytes()


def main():
    scaler, pca, kmeans, clmap = load_bundle_and_mapping()

    sg.theme("SystemDefaultForReal")
    col_left = [
        [sg.Text("Agente IA – Piezas", font=("Segoe UI", 16, "bold"))],
        [sg.Button("Ejecutar (10 aleatorias)", key="-RUN10-", size=(24,1))],
        [sg.Button("Escuchar", key="-LISTEN-", size=(24,1))],
        [sg.Button("Subir imagen…", key="-UPLOAD-", size=(24,1))],
        [sg.Button("Salir", key="-EXIT-", size=(24,1), button_color=("white","firebrick3"))],
        [sg.Text("Última imagen:", pad=(0,(12,3)))],
        [sg.Image(key="-IMG-", size=(380,380), background_color="#f0f0f0", pad=(0,10))],
    ]

    col_right = [
        [sg.Text("Log", font=("Segoe UI", 12, "bold"))],
        [sg.Multiline(size=(70,28), key="-LOG-", autoscroll=True, disabled=True, font=("Consolas", 10))]
    ]

    layout = [
        [sg.Column(col_left, pad=(10,10)), sg.VSeperator(), sg.Column(col_right, pad=(10,10))]
    ]
    win = sg.Window("Agente IA – Vision + Voz + Bayes", layout, finalize=True)

    last_counts = {}  # último conteo de RUN10 para Bayes

    def log(msg):
        win["-LOG-"].update(f"{msg}\n", append=True)

    log("Cargado bundle y mapping. Clases esperadas para Bayes (orden): " + ", ".join(CLASS_ORDER))

    while True:
        ev, vals = win.read()
        if ev in (sg.WIN_CLOSED, "-EXIT-"):
            break

        if ev == "-RUN10-":
            try:
                counts, per_image = sample_and_classify_10(scaler, pca, kmeans, clmap)
                last_counts = counts
                log("== Muestreo de 10 ==")
                for p, name in per_image:
                    log(f"- {os.path.basename(p)} -> {name}")
                log("Conteo total:")
                for k in sorted(counts.keys()):
                    log(f"  {k:10s}: {counts[k]}")
                log("")
            except Exception as e:
                log(f"[ERROR] {e}")

        if ev == "-LISTEN-":
            cmd = (listen_command() or "").strip().lower()
            if not cmd:
                log("[AVISO] No recibí comando.")
                continue
            log(f"[Comando] {cmd}")

            if cmd in ("proporcion", "proporción"):
                if not last_counts:
                    log("[AVISO] Primero usá 'Ejecutar (10 aleatorias)' para tener un conteo.")
                    continue
                try:
                    vec, posterior, exp_prop, caja = run_bayes(last_counts)
                    log("Vector (tornillos, clavos, arandelas, tuercas) = " + str(vec))
                    log("Posterior por caja (A,B,C,D) = " + np.array2string(np.array(posterior), precision=3))
                    log("Proporciones esperadas por clase = " + np.array2string(np.array(exp_prop), precision=3))
                    approx = [int(round(1000*p)) for p in exp_prop]
                    log("≈ Cantidades en 1000: " + ", ".join(f"{c}:{n}" for c,n in zip(CLASS_ORDER, approx)))
                    log(f"→ Caja MAP: {caja}")
                    log("")
                except Exception as e:
                    log(f"[ERROR Bayes] {e}")

            elif cmd == "contar":
                if not last_counts:
                    log("[AVISO] No hay conteo aún. Ejecutá primero el muestreo.")
                else:
                    log("== Conteo (última muestra de 10) ==")
                    for k in CLASS_ORDER:
                        log(f"  {k:10s}: {last_counts.get(k,0)}")
                    otros = {k:v for k,v in last_counts.items() if k not in CLASS_ORDER}
                    for k,v in otros.items():
                        log(f"  {k:10s}: {v}")
                    log("")
            elif cmd == "salir":
                break
            else:
                log("Comando no reconocido. Usá: proporcion / contar / salir")

        if ev == "-UPLOAD-":
            path = sg.popup_get_file("Elegí una imagen", file_types=(("Imágenes", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp"),))
            if not path:
                continue
            try:
                lab, name = classify_image(path, scaler, pca, kmeans, clmap, save_debug=True)
                feat, mask, img = extract_features_img(path)
                win["-IMG-"].update(data=im_to_bytes(img))
                log(f"[Subida] {os.path.basename(path)} -> cluster {lab} / etiqueta: {name}")
                base = os.path.splitext(os.path.basename(path))[0]
                log(f"Se guardó máscara: debug_mask_{base}.png")
                log("")
            except Exception as e:
                log(f"[ERROR] {e}")

    win.close()


if __name__ == "__main__":
    main()
