#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agente IA – Demo Aleatoria + Subir imagen + Voz
- "Elegir 10 al azar": muestreo desde dataset_mix con extractor ORIGINAL (igual al entrenamiento del joblib)
- "Subir imagen...": clasifica 1 imagen usando extractor ROBUSTO (mejor para fotos nuevas)
- Bayes usa solo los conteos devueltos por visión (no altera la clasificación)
Requisitos (pip):
  PySimpleGUI opencv-python scikit-image scikit-learn joblib numpy soundfile librosa sounddevice
"""

import json, random
from pathlib import Path
import numpy as np
import PySimpleGUI as sg
import cv2
from joblib import load
from skimage.feature import local_binary_pattern
import soundfile as sf
import librosa
import sounddevice as sd

# -------- Rutas por defecto (cambiables desde la GUI) ----------
KMEANS_MODEL_PATH = "kmeans_piezas.joblib"
VOICE_MODEL_PATH  = "knn_voice.joblib"
DATASET_DIR       = "dataset_mix"

# -------- Vocabulario canónico y aliases -----------------------
CLASSES_ORDER = ["tornillos","clavos","arandelas","tuercas"]
ALIASES = {
    "tornillo":"tornillos","tornillos":"tornillos",
    "clavo":"clavos","clavos":"clavos",
    "arandela":"arandelas","arandelas":"arandelas",
    "tuerca":"tuercas","tuercas":"tuercas",
}

# -------- Audio ------------------------------------------------
MIC_SECONDS = 1.5
MIC_RATE    = 16000

# -------- Utilidades ------------------------------------------
def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    paths=[]
    for e in exts: paths += list(Path(folder).glob(e))
    return sorted([str(p) for p in paths])

# ================== Helper de preprocesado =====================
def _gray_world_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1,3).mean(axis=0) + 1e-6
    gray  = means.mean()
    gain  = gray / means
    img  *= gain
    return np.clip(img,0,255).astype(np.uint8)

def _auto_gamma(img_bgr, target_mean=0.6):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V   = hsv[...,2].astype(np.float32)/255.0
    m   = max(V.mean(), 1e-6)
    gamma = np.log(target_mean)/np.log(m)
    Vc  = np.clip(V**gamma, 0, 1)
    hsv[...,2] = (Vc*255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _make_mask_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V   = hsv[...,2]
    V   = cv2.equalizeHist(V)
    bw  = cv2.adaptiveThreshold(V,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,31,5)
    bw  = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8),1)
    bw  = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),1)
    return bw

def _largest_contour_crop(img_bgr, mask_bw, pad_ratio=0.10):
    cnts,_ = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img_bgr, None
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    pad = int(max(w,h)*pad_ratio)
    x = max(x-pad,0); y = max(y-pad,0)
    H,W = img_bgr.shape[:2]
    w = min(w+2*pad, W-x); h = min(h+2*pad, H-y)
    return img_bgr[y:y+h, x:x+w], c

# ================== Extractor ORIGINAL (bundle) =================
def extract_features_image_original(path):
    """
    Igual al usado para entrenar el joblib: 7 Hu + 10 LBP + 512 HSV (529 dims).
    Sin recortes especiales ni balances extra (salvo equalize + blur + adaptiveThreshold).
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo leer {path}")
    img  = cv2.resize(img, (256,256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    mask = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,31,5)
    # Hu
    m  = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-12)
    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    n_bins = 10
    lbp_hist,_ = np.histogram(lbp.ravel(), bins=n_bins, range=(0,n_bins), density=True)
    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv],[0,1,2], None, (8,8,8), [0,180,0,256,0,256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()
    return np.hstack([hu, lbp_hist, hsv_hist]).astype(np.float32)

# ================== Extractor ROBUSTO (fotos nuevas) ============
def extract_features_image_robust(path):
    """
    Robusto para fotos del celu (sin reentrenar):
      - Gray-World + auto-gamma
      - Contraste (alpha=1.4, beta=20)
      - Máscara HSV + morfología
      - Recorte por contorno principal
      - Hu (con máscara), LBP y HSV en región del objeto
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo leer {path}")

    img = _gray_world_balance(img)
    img = _auto_gamma(img, target_mean=0.6)

    # trabajar grande para segmentar
    img = cv2.resize(img, (512,512))
    mask = _make_mask_hsv(img)
    crop, c = _largest_contour_crop(img, mask, pad_ratio=0.10)
    if c is None:
        crop = img

    crop = cv2.resize(crop, (256,256))
    # contraste y resegmentación
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=20)
    gray = cv2.equalizeHist(gray)
    mask2 = _make_mask_hsv(crop)
    if mask2.mean() < 10:
        crop  = _auto_gamma(crop, target_mean=0.7)
        mask2 = _make_mask_hsv(crop)

    # Hu con máscara
    m  = cv2.moments(mask2)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-12)

    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    n_bins = 10
    lbp_hist,_ = np.histogram(lbp.ravel(), bins=n_bins, range=(0,n_bins), density=True)

    # HSV (solo objeto si hay área suficiente)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_bool = (mask2>0).astype(np.uint8)
    if mask_bool.sum() > 100:
        hist = cv2.calcHist([hsv],[0,1,2], mask_bool, (8,8,8), [0,180,0,256,0,256])
    else:
        hist = cv2.calcHist([hsv],[0,1,2], None,       (8,8,8), [0,180,0,256,0,256])
    hsv_hist = cv2.normalize(hist, hist).flatten()

    return np.hstack([hu, lbp_hist, hsv_hist]).astype(np.float32)

# ================== KMeans helpers ==============================
def _label_from_cluster(clus, cluster_to_class, class_names):
    raw = cluster_to_class.get(clus, None)
    if isinstance(raw,int) and 0 <= raw < len(class_names):
        raw = class_names[raw]
    if not raw: return None
    raw = str(raw).strip().lower()
    if raw in ALIASES: return ALIASES[raw]
    for std in CLASSES_ORDER:
        if std in raw or raw in std:
            return std
    return None

def classify_one_image(path, model_path):
    b = load(model_path)
    kmeans, scaler = b["kmeans"], b["scaler"]
    pca = b.get("pca", None)
    c2c = b["cluster_to_class"]; classes = b["classes"]
    x  = extract_features_image_robust(path).reshape(1,-1)   # ROBUSTO para fotos nuevas
    xs = scaler.transform(x)
    if pca is not None: xs = pca.transform(xs)
    clus = int(kmeans.predict(xs)[0])
    label = _label_from_cluster(clus, c2c, classes)
    if label is None: label = f"cluster_{clus}"
    # --- Diagnóstico rápido de la máscara en el robusto ---
    xdbg = extract_features_image_robust(path)  # usa adentro la máscara
    # Recomputamos solo la máscara para medir cobertura:
    img_dbg = cv2.imread(path); img_dbg = cv2.resize(img_dbg, (512,512))
    mask_dbg = _make_mask_hsv(img_dbg)
    area = (mask_dbg > 0).mean() * 100   # % de pixeles en la máscara
    print(f"[DEBUG] mask coverage: {area:.1f}%  (esperable >15-20%)")
    return label, clus

def classify_random_10(dataset_dir, model_path):
    imgs = list_images(dataset_dir)
    if len(imgs) < 10:
        raise SystemExit(f"Necesito >=10 imágenes en '{dataset_dir}'. Hay {len(imgs)}.")
    sel = random.sample(imgs, 10)

    b = load(model_path)
    kmeans, scaler = b["kmeans"], b["scaler"]
    pca = b.get("pca", None)
    c2c = b["cluster_to_class"]; classes = b["classes"]

    counts = {k:0 for k in CLASSES_ORDER}
    unknown = 0
    cluster_counts = {}

    for p in sel:
        x  = extract_features_image_original(p).reshape(1,-1)  # ORIGINAL para la muestra
        xs = scaler.transform(x)
        if pca is not None: xs = pca.transform(xs)
        clus = int(kmeans.predict(xs)[0])
        cluster_counts[clus] = cluster_counts.get(clus,0)+1
        label = _label_from_cluster(clus, c2c, classes)
        if label is None: unknown += 1
        else: counts[label]+=1

    Path("muestra10_seleccion.txt").write_text("\n".join(sel), encoding="utf-8")
    Path("sample_counts.json").write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    Path("diagnostico_clusters.json").write_text(
        json.dumps({"cluster_counts": cluster_counts, "unknown": unknown}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return sel, [counts.get(k,0) for k in CLASSES_ORDER]

# ================== Bayes (proporción) =========================
def posterior_over_boxes(counts):
    BOXES = {
        "a":[0.25,0.25,0.25,0.25],
        "b":[0.15,0.30,0.30,0.25],
        "c":[0.25,0.35,0.25,0.15],
        "d":[0.50,0.50,0.00,0.00],
    }
    PRIOR = {k:1/len(BOXES) for k in BOXES}
    eps = 1e-9
    logs={}
    for name, probs in BOXES.items():
        ll=0.0
        for c,p in zip(counts, probs):
            p=max(p,eps)
            if c>0: ll += c*np.log(p)
        logs[name]=np.log(PRIOR[name])+ll
    m=max(logs.values())
    exps={k:np.exp(v-m) for k,v in logs.items()}
    Z=sum(exps.values())
    post={k:v/Z for k,v in exps.items()}
    exp=[sum(post[b]*BOXES[b][i] for b in BOXES) for i in range(4)]
    pred1000=[int(round(1000*p)) for p in exp]
    return post, exp, pred1000

# ================== Voz (KNN) =================================
def extract_features_audio(y, sr):
    target_len = int(0.8*sr)
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1   = librosa.feature.delta(mfcc, order=1)
    d2   = librosa.feature.delta(mfcc, order=2)
    def stats(M): return np.hstack([M.mean(axis=1), M.std(axis=1)])
    feat = np.hstack([
        stats(mfcc), stats(d1), stats(d2),
        librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean(),
        librosa.feature.zero_crossing_rate(y).mean(),
    ])
    return feat.astype(np.float32)

def predict_command_from_mic(clf):
    sg.popup_quick_message("Grabando...", auto_close_duration=1, background_color="yellow")
    rec = sd.rec(int(MIC_SECONDS*MIC_RATE), samplerate=MIC_RATE, channels=1, dtype="float32")
    sd.wait()
    y = rec.squeeze()
    x = extract_features_audio(y, MIC_RATE).reshape(1,-1)
    return clf.predict(x)[0]

def predict_command_from_wav(clf, wav_path):
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim>1: y = y.mean(axis=1)
    if sr != MIC_RATE: y = librosa.resample(y, orig_sr=sr, target_sr=MIC_RATE)
    x = extract_features_audio(y, MIC_RATE).reshape(1,-1)
    return clf.predict(x)[0]

# ================== GUI =======================================
def main():
    sg.theme("SystemDefault")

    layout = [
        [sg.Text("Agente IA — Demo Aleatoria + Subir imagen", font=("Segoe UI", 13, "bold"))],
        [sg.Text("K-Means:"), sg.Input(KMEANS_MODEL_PATH, key="-KMEANS-", size=(42,1)), sg.FileBrowse("Buscar")],
        [sg.Text("Voz:"),     sg.Input(VOICE_MODEL_PATH,  key="-VOICE-",  size=(42,1)), sg.FileBrowse("Buscar")],
        [sg.Text("Dataset:"), sg.Input(DATASET_DIR,       key="-DATASET-", size=(42,1)), sg.FolderBrowse("Elegir")],
        [sg.HorizontalSeparator()],
        [sg.Button("🎲 Elegir 10 al azar", key="-RUN-"),
         sg.Button("🎤 Escuchar (mic)",   key="-MIC-"),
         sg.Button("🎧 Elegir WAV…",      key="-WAV-"),
         sg.Button("📂 Subir imagen…",    key="-UPLOAD-"),
         sg.Button("Salir",               key="-EXIT-")],
        [sg.HorizontalSeparator()],
        [sg.Text("Salida:")],
        [sg.Multiline("", size=(100,20), key="-OUT-", autoscroll=True, font=("Consolas", 10))],
    ]
    win = sg.Window("Agente IA - Desktop", layout, finalize=True)
    log = lambda m: win["-OUT-"].print(m)

    clf_voice = None

    while True:
        ev, vals = win.read()
        if ev in (sg.WINDOW_CLOSED, "-EXIT-"):
            break

        if ev == "-RUN-":
            try:
                kpath = Path(vals["-KMEANS-"]).resolve()
                log(f"[DIAG] Modelo KMeans: {kpath}")
                b = load(str(kpath))
                log("[DIAG] classes: " + str(b.get("classes")))
                log("[DIAG] cluster_to_class: " + str(b.get("cluster_to_class")))
                sel, vec = classify_random_10(vals["-DATASET-"], str(kpath))
                log("[VISIÓN] Conteos (10): " + str(dict(zip(CLASSES_ORDER, vec))))
                log("Rutas guardadas en 'muestra10_seleccion.txt'.")
                try:
                    diag = json.loads(Path("diagnostico_clusters.json").read_text(encoding="utf-8"))
                    log("[DIAG] clusters en muestra: " + str(diag.get("cluster_counts")))
                    if diag.get("unknown",0)>0:
                        log(f"[ALERTA] {diag['unknown']} imagen(es) sin mapeo/alias.")
                except Exception as e:
                    log(f"[DIAG] No se pudo leer diagnóstico: {e}")
            except Exception as e:
                log(f"[ERROR VISIÓN] {e}")

        if ev == "-MIC-":
            try:
                if clf_voice is None:
                    vpath = Path(vals["-VOICE-"]).resolve()
                    bundle_v = load(str(vpath))
                    clf_voice = bundle_v["pipeline"]
                    log(f"✅ Modelo de voz cargado: {vpath.name}")
                pred = predict_command_from_mic(clf_voice)
                log(f"[VOZ] Comando: {pred}")
                cmd = str(pred).lower()
                if cmd.startswith("cont"):
                    data = json.loads(Path("sample_counts.json").read_text(encoding="utf-8")) \
                           if Path("sample_counts.json").exists() else None
                    if not data: log("No hay conteos. Usá 'Elegir 10 al azar'.")
                    else: log(f"[ACCION] Conteos → {dict(zip(CLASSES_ORDER, [int(data.get(k,0)) for k in CLASSES_ORDER]))}")
                elif cmd.startswith("prop"):
                    data = json.loads(Path("sample_counts.json").read_text(encoding="utf-8")) \
                           if Path("sample_counts.json").exists() else None
                    if not data: log("No hay conteos. Usá 'Elegir 10 al azar'.")
                    else:
                        counts = [int(data.get(k,0)) for k in CLASSES_ORDER]
                        post, exp, pred1000 = posterior_over_boxes(counts)
                        log("[ACCION] Proporciones bayesianas:")
                        for name,p,n in zip(CLASSES_ORDER, exp, pred1000):
                            log(f"  {name:<10} {p*100:6.2f}%  ({n}/1000)")
                        log(f"  Caja MAP: {max(post, key=post.get)}")
                elif cmd.startswith("sal"):
                    log("[ACCION] Finalizar"); break
                else:
                    log("Comando no reconocido.")
            except Exception as e:
                log(f"[ERROR VOZ] {e}")

        if ev == "-WAV-":
            path = sg.popup_get_file("Elegí un WAV", file_types=(("WAV","*.wav"),))
            if path:
                try:
                    if clf_voice is None:
                        vpath = Path(vals["-VOICE-"]).resolve()
                        bundle_v = load(str(vpath))
                        clf_voice = bundle_v["pipeline"]
                        log(f"✅ Modelo de voz cargado: {vpath.name}")
                    pred = predict_command_from_wav(clf_voice, path)
                    log(f"[VOZ] Comando (wav): {pred}")
                except Exception as e:
                    log(f"[ERROR VOZ] {e}")

        if ev == "-UPLOAD-":
            img_path = sg.popup_get_file(
                "Elegí una imagen",
                file_types=(("Imagen","*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp"),),
            )
            if img_path:
                try:
                    kpath = Path(vals["-KMEANS-"]).resolve()
                    label, clus = classify_one_image(img_path, str(kpath))
                    log(f"[UPLOAD] {Path(img_path).name} → {label} (cluster {clus})")
                except Exception as e:
                    log(f"[ERROR UPLOAD] {e}")

    win.close()

if __name__ == "__main__":
    main()
