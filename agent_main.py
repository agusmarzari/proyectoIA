#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, json, random, time
from pathlib import Path
import numpy as np

# --- GUI ---
import PySimpleGUI as sg

# --- Audio & ML ---
import soundfile as sf
import sounddevice as sd
import librosa
from joblib import load

# --- Vision (OpenCV + skimage) ---
import cv2
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler

# -------------------------------
# CONFIG
# -------------------------------
VOICE_MODEL_PATH = "knn_voice.joblib"          # generado por voice_knn.py
KMEANS_MODEL_PATH = "kmeans_piezas.joblib"     # generado por kmeans_mejorado.py (bundle con scaler, mapping, etc.)
MUESTRA_DIR = "muestra10"                       # carpeta con 10 imágenes para clasificar
CLASSES_ORDER = ["tornillos","clavos","arandelas","tuercas"]  # orden fijo para Bayes
MIC_SECONDS = 1.5
MIC_RATE = 16000

# -------------------------------
# Feature extraction (matching training)
# -------------------------------
LBP_P, LBP_R = 8, 1
LBP_METHOD = "uniform"
HSV_BINS = (8, 8, 8)

def preprocess_mask(gray):
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 5)
    return bw

def hu_moments_from_mask(mask):
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu

def lbp_hist(gray):
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def hsv_hist(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,HSV_BINS,[0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo leer {path}")
    img = cv2.resize(img, (256,256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = preprocess_mask(gray)
    hu = hu_moments_from_mask(mask)
    lbp = lbp_hist(gray)
    hsv = hsv_hist(img)
    return np.hstack([hu, lbp, hsv]).astype(np.float32)

# -------------------------------
# Voice features (matching voice_knn.py)
# -------------------------------
def extract_features_audio(y, sr):
    target_len = int(0.8 * sr)
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(mat):
        return np.hstack([mat.mean(axis=1), mat.std(axis=1)])

    feat_mfcc = stats(mfcc)
    feat_d1 = stats(d1)
    feat_d2 = stats(d2)

    S, phase = librosa.magphase(librosa.stft(y, n_fft=512, hop_length=160, win_length=400))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    feat = np.hstack([feat_mfcc, feat_d1, feat_d2, centroid, rolloff, zcr])
    return feat.astype(np.float32)

# -------------------------------
# Bayesian estimator (inline)
# -------------------------------
BOXES = {
    "a": [0.25, 0.25, 0.25, 0.25],
    "b": [0.15, 0.30, 0.30, 0.25],
    "c": [0.25, 0.35, 0.25, 0.15],
    "d": [0.50, 0.50, 0.00, 0.00],
}
PRIOR = {k: 1.0/len(BOXES) for k in BOXES}
EPS = 1e-9

def log_likelihood(counts, probs):
    ll = 0.0
    for c, p in zip(counts, probs):
        p_eff = max(p, EPS)
        if c == 0: 
            continue
        ll += c * np.log(p_eff)
    return float(ll)

def posterior_over_boxes(counts):
    logs = {}
    for name, probs in BOXES.items():
        ll = log_likelihood(counts, probs)
        logs[name] = np.log(PRIOR[name]) + ll
    m = max(logs.values())
    exps = {k: np.exp(v - m) for k, v in logs.items()}
    Z = sum(exps.values())
    return {k: v/Z for k,v in exps.items()}

def expected_proportions(posterior):
    exp = [0.0]*4
    for name, w in posterior.items():
        probs = BOXES[name]
        for i in range(4):
            exp[i] += w * probs[i]
    return exp

# -------------------------------
# Vision: classify folder of 10 images
# -------------------------------
def classify_muestra10(kmeans_bundle_path, muestra_dir):
    if not Path(kmeans_bundle_path).exists():
        raise SystemExit("No se encontró kmeans_piezas.joblib. Entrena/coloca el modelo de visión en la carpeta del programa.")
    bundle = load(kmeans_bundle_path)
    kmeans = bundle["kmeans"]
    scaler = bundle["scaler"]
    cluster_to_class = bundle["cluster_to_class"]
    class_names = bundle["classes"]  # ["tornillos","tuercas","arandelas","clavos"] (según como lo entrenaste)

    # Recolectar 10 imágenes
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(muestra_dir, e)))
    if len(paths) != 10:
        raise SystemExit(f"Se esperaban 10 imágenes en '{muestra_dir}', encontradas: {len(paths)}.")

    counts = {name:0 for name in CLASSES_ORDER}
    for p in paths:
        feat = extract_features_image(p).reshape(1,-1)
        fs = scaler.transform(feat)
        clus = int(kmeans.predict(fs)[0])
        clazz_idx = cluster_to_class.get(clus, None)
        clazz = class_names[clazz_idx] if clazz_idx is not None else None
        # Mapear al orden estándar (CLASSES_ORDER)
        # Ajuste: los nombres en class_names deben coincidir con nuestros nombres esperados
        if clazz is None:
            continue
        # Normalizar nombre para evitar singular/plural
        key = None
        for std in CLASSES_ORDER:
            if std.lower() in clazz.lower() or clazz.lower() in std.lower():
                key = std
                break
        if key is None:
            # si no matchea, intentalo tal cual
            key = clazz
        if key not in counts:
            # si aún no, crea entrada
            counts[key] = 0
        counts[key] += 1

    # Asegurar orden
    return [counts.get(k,0) for k in CLASSES_ORDER], counts

# -------------------------------
# Voice: predict command
# -------------------------------
def predict_command_from_mic(model_path, seconds=MIC_SECONDS, rate=MIC_RATE):
    if not Path(model_path).exists():
        raise SystemExit("No se encontró knn_voice.joblib. Entrena/coloca el modelo de voz en la carpeta del programa.")
    bundle = load(model_path)
    clf = bundle["pipeline"]
    sg.popup_quick_message("Grabando comando...", auto_close_duration=1, background_color='yellow')
    rec = sd.rec(int(seconds*rate), samplerate=rate, channels=1, dtype='float32')
    sd.wait()
    y = rec.squeeze()
    feat = extract_features_audio(y, rate).reshape(1, -1)
    pred = clf.predict(feat)[0]
    return pred

def predict_command_from_wav(model_path, wav_path):
    if not Path(model_path).exists():
        raise SystemExit("No se encontró knn_voice.joblib.")
    bundle = load(model_path)
    clf = bundle["pipeline"]
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != MIC_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=MIC_RATE)
        sr = MIC_RATE
    feat = extract_features_audio(y, sr).reshape(1, -1)
    pred = clf.predict(feat)[0]
    return pred

# -------------------------------
# GUI App
# -------------------------------
def main():
    sg.theme("SystemDefault")
    layout = [
        [sg.Text("Agente IA: Visión (K-Means) + Bayes + Voz (K-NN)", font=("Segoe UI", 14, "bold"))],
        [sg.Text("Modelos:"), sg.Input(KMEANS_MODEL_PATH, key="-KMEANS-", size=(40,1)), sg.FileBrowse("Buscar")],
        [sg.Text(" "), sg.Input(VOICE_MODEL_PATH, key="-VOICE-", size=(40,1)), sg.FileBrowse("Buscar")],
        [sg.Text("Carpeta muestra (10 imágenes):"), sg.Input(MUESTRA_DIR, key="-MUESTRA-", size=(40,1)), sg.FolderBrowse("Elegir")],
        [sg.HorizontalSeparator()],
        [sg.Button("Ejecutar (tomar muestra de 10)", key="-RUNVISION-", button_color=("white","#0078D7")),
         sg.Button("Escuchar comando (mic)", key="-MIC-"),
         sg.Button("Elegir WAV…", key="-WAV-"),
         sg.Button("Salir", key="-EXIT-")],
        [sg.HorizontalSeparator()],
        [sg.Text("Salida:")],
        [sg.Multiline("", size=(100,20), key="-OUT-", autoscroll=True, font=("Consolas", 10))],
    ]
    win = sg.Window("Agente IA - Demostrador", layout, finalize=True)

    sample_counts = None  # [tornillos, clavos, arandelas, tuercas]

    def log(msg):
        win["-OUT-"].print(msg)

    while True:
        ev, vals = win.read()
        if ev in (sg.WINDOW_CLOSED, "-EXIT-"):
            break

        if ev == "-RUNVISION-":
            try:
                counts_vec, counts_map = classify_muestra10(vals["-KMEANS-"], vals["-MUESTRA-"])
                sample_counts = counts_vec
                log(f"[VISIÓN] Conteos muestra (10): {dict(zip(CLASSES_ORDER, counts_vec))}")
                # guardar json
                Path("sample_counts.json").write_text(json.dumps(dict(zip(CLASSES_ORDER, counts_vec)), indent=2), encoding="utf-8")
                log("[VISIÓN] Guardado sample_counts.json")
            except Exception as e:
                log(f"[ERROR VISIÓN] {e}")

        if ev == "-MIC-":
            try:
                pred = predict_command_from_mic(vals["-VOICE-"])
                log(f"[VOZ] Comando detectado (mic): {pred}")
                # Ejecutar acción
                if pred.lower().startswith("cont"):
                    if sample_counts is None:
                        # intenta leer del json
                        if Path("sample_counts.json").exists():
                            data = json.loads(Path("sample_counts.json").read_text(encoding="utf-8"))
                            sample_counts = [int(data.get(k,0)) for k in CLASSES_ORDER]
                    if sample_counts is None:
                        log("[VOZ] No hay conteos disponibles aún. Primero ejecuta 'Ejecutar (tomar muestra)'.")
                    else:
                        log(f"[ACCION] Conteos → {dict(zip(CLASSES_ORDER, sample_counts))}")
                elif pred.lower().startswith("prop"):
                    if sample_counts is None and Path("sample_counts.json").exists():
                        data = json.loads(Path("sample_counts.json").read_text(encoding="utf-8"))
                        sample_counts = [int(data.get(k,0)) for k in CLASSES_ORDER]
                    if sample_counts is None:
                        log("[VOZ] No hay conteos disponibles. Primero ejecuta 'Ejecutar (tomar muestra)'.")
                    else:
                        post = posterior_over_boxes(sample_counts)
                        map_box = max(post.items(), key=lambda kv: kv[1])[0]
                        exp_p = expected_proportions(post)
                        pred1000 = [int(round(1000*p)) for p in exp_p]
                        log("[ACCION] Proporciones bayesianas:")
                        for name, p, n in zip(CLASSES_ORDER, exp_p, pred1000):
                            log(f"  {name:<10}  {p*100:6.2f}%   ({n}/1000)")
                        log(f"  Caja MAP: {map_box}")
                elif pred.lower().startswith("sal"):
                    log("[ACCION] Finalizar")
                    break
                else:
                    log("[VOZ] Comando no reconocido. Usa 'proporcion', 'contar' o 'salir'.")
            except Exception as e:
                log(f"[ERROR VOZ] {e}")

        if ev == "-WAV-":
            wav_path = sg.popup_get_file("Elegí un WAV con el comando", file_types=(("WAV","*.wav"),), no_window=True)
            if wav_path:
                try:
                    pred = predict_command_from_wav(vals["-VOICE-"], wav_path)
                    log(f"[VOZ] Comando detectado (wav): {pred}")
                    # Igual que mic
                    if pred.lower().startswith("cont"):
                        if sample_counts is None and Path("sample_counts.json").exists():
                            data = json.loads(Path("sample_counts.json").read_text(encoding="utf-8"))
                            sample_counts = [int(data.get(k,0)) for k in CLASSES_ORDER]
                        if sample_counts is None:
                            log("[VOZ] No hay conteos disponibles aún. Primero ejecuta 'Ejecutar (tomar muestra)'.")
                        else:
                            log(f"[ACCION] Conteos → {dict(zip(CLASSES_ORDER, sample_counts))}")
                    elif pred.lower().startswith("prop"):
                        if sample_counts is None and Path("sample_counts.json").exists():
                            data = json.loads(Path("sample_counts.json").read_text(encoding="utf-8"))
                            sample_counts = [int(data.get(k,0)) for k in CLASSES_ORDER]
                        if sample_counts is None:
                            log("[VOZ] No hay conteos disponibles. Primero ejecuta 'Ejecutar (tomar muestra)'.")
                        else:
                            post = posterior_over_boxes(sample_counts)
                            map_box = max(post.items(), key=lambda kv: kv[1])[0]
                            exp_p = expected_proportions(post)
                            pred1000 = [int(round(1000*p)) for p in exp_p]
                            log("[ACCION] Proporciones bayesianas:")
                            for name, p, n in zip(CLASSES_ORDER, exp_p, pred1000):
                                log(f"  {name:<10}  {p*100:6.2f}%   ({n}/1000)")
                            log(f"  Caja MAP: {map_box}")
                    elif pred.lower().startswith("sal"):
                        log("[ACCION] Finalizar")
                        break
                    else:
                        log("[VOZ] Comando no reconocido. Usa 'proporcion', 'contar' o 'salir'.")
                except Exception as e:
                    log(f"[ERROR VOZ] {e}")

    win.close()

if __name__ == "__main__":
    main()
