OVERRIDE_PATH = "cluster_override.json"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agente IA - GUI (PySimpleGUI)
- Seleccionar 10 imágenes al azar del dataset y clasificarlas con K-Means (.joblib)
- Subir UNA imagen y clasificarla (mismo pipeline)
- Escuchar comando por mic (contar / proporción / salir) con KNN de voz (.joblib)
- Elegir WAV para comando
- Diagnóstico: imprime ruta del modelo, classes, cluster_to_class y clusters observados
- Opción (checkbox) para probar recorte por contorno, OFF por defecto (mantiene pipeline original)

Requisitos:
  pip install PySimpleGUI opencv-python scikit-image scikit-learn joblib numpy soundfile librosa sounddevice
"""

import json, random
from pathlib import Path
import numpy as np
import PySimpleGUI as sg
import cv2
from skimage.feature import local_binary_pattern
from joblib import load
import soundfile as sf
import librosa
import sounddevice as sd

# ---------------- Config ----------------
# Por tu comentario, los .joblib están junto al script:
VOICE_MODEL_PATH = "knn_voice.joblib"
KMEANS_MODEL_PATH = "kmeans_piezas.joblib"
DATASET_DIR = "dataset_mix"

# Nombres canónicos (para normalizar etiquetas)
CLASSES_ORDER = ["tornillos", "clavos", "arandelas", "tuercas"]

# Aliases (singular/plural/tildes) → nombre canónico
ALIASES = {
    "tornillo": "tornillos", "tornillos": "tornillos",
    "clavo": "clavos", "clavos": "clavos",
    "arandela": "arandelas", "arandelas": "arandelas",
    "tuerca": "tuercas", "tuercas": "tuercas",
}

# Audio
MIC_SECONDS = 1.5
MIC_RATE = 16000

# Features (pipeline original)
LBP_P, LBP_R = 8, 1
LBP_METHOD = "uniform"
HSV_BINS = (8, 8, 8)  # 512 bins

# Flag global (se controla con checkbox en la GUI)
USE_CROP = False  # OFF por defecto para mantener coherencia con el joblib original

# --------------- Utilidades ---------------
def list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths = []
    for e in exts:
        paths.extend(Path(folder).glob(e))
    return sorted([str(p) for p in paths])

# ----------- Extractores de features -----------
def _crop_by_largest_contour(img_bgr, mask_bw, pad_ratio=0.10):
    cnts, _ = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = int(max(w, h) * pad_ratio)
    x = max(x - pad, 0); y = max(y - pad, 0)
    H, W = img_bgr.shape[:2]
    w = min(w + 2*pad, W - x); h = min(h + 2*pad, H - y)
    return img_bgr[y:y+h, x:x+w]

def extract_features_image(path):
    """
    PIPELINE ORIGINAL (el que entrenó el K-Means): SIN recorte por contorno.
    7(Hu) + 10(LBP) + 512(HSV) = 529 dims.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo leer {path}")
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
    mask = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    # Hu moments
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    # LBP uniform
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # HSV hist
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0,1,2], None, HSV_BINS, [0,180,0,256,0,256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    feat = np.hstack([hu, lbp_hist, hsv_hist]).astype(np.float32)
    return feat

def extract_features_image_cropped(path):
    """
    Variante con recorte por mayor contorno (OPCIONAL).
    Útil si querés robustecer a mucho fondo, pero puede desalinear un joblib entrenado sin recorte.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo leer {path}")
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    img = _crop_by_largest_contour(img, bw, pad_ratio=0.10)

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
    mask = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0,1,2], None, HSV_BINS, [0,180,0,256,0,256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    feat = np.hstack([hu, lbp_hist, hsv_hist]).astype(np.float32)
    return feat

def get_extractor():
    return extract_features_image_cropped if USE_CROP else extract_features_image

# ---------------- Clasificación (KMeans bundle) ----------------
def _label_from_cluster(clus, cluster_to_class, class_names):
    raw = cluster_to_class.get(clus, None)
    # Si viene como índice numérico, convertirlo a string
    if isinstance(raw, int) and 0 <= raw < len(class_names):
        raw = class_names[raw]
    if not raw:
        return None  # sin mapeo → desconocido

    raw = str(raw).strip().lower()

    # Normalización por alias (singular/plural)
    if raw in ALIASES:
        return ALIASES[raw]

    # Intento robusto de match por inclusión
    for std in CLASSES_ORDER:
        if std in raw or raw in std:
            return std

    return None  # no pudimos normalizar → desconocido

def classify_one_image(path, model_path):
    bundle = load(model_path)
    # Aplicar override si existe
    try:
        if Path(OVERRIDE_PATH).exists():
            override = json.loads(Path(OVERRIDE_PATH).read_text(encoding="utf-8"))
            if isinstance(override, dict) and override:
                bundle["cluster_to_class"] = {int(k): v for k, v in override.items()}
    except Exception as _e:
        pass

    kmeans = bundle["kmeans"]
    scaler = bundle["scaler"]
    pca = bundle.get("pca", None)
    cluster_to_class = bundle["cluster_to_class"]
    class_names = bundle["classes"]

    extractor = get_extractor()
    feat = extractor(path).reshape(1, -1)
    fs = scaler.transform(feat)
    if pca is not None:
        fs = pca.transform(fs)

    clus = int(kmeans.predict(fs)[0])
    label = _label_from_cluster(clus, cluster_to_class, class_names)
    return label, clus

def classify_random_10(dataset_dir, model_path):
    imgs = list_images(dataset_dir)
    if len(imgs) < 10:
        raise SystemExit(f"Necesito >=10 imágenes en '{dataset_dir}'. Hay {len(imgs)}.")
    sel = random.sample(imgs, 10)

    bundle = load(model_path)
    # Aplicar override si existe
    try:
        if Path(OVERRIDE_PATH).exists():
            override = json.loads(Path(OVERRIDE_PATH).read_text(encoding="utf-8"))
            if isinstance(override, dict) and override:
                bundle["cluster_to_class"] = {int(k): v for k, v in override.items()}
    except Exception as _e:
        pass
    kmeans = bundle["kmeans"]
    scaler = bundle["scaler"]
    pca = bundle.get("pca", None)
    cluster_to_class = bundle["cluster_to_class"]
    class_names = bundle["classes"]

    counts = {k: 0 for k in CLASSES_ORDER}
    unknown = 0
    cluster_counts = {}
    extractor = get_extractor()

    for p in sel:
        feat = extractor(p).reshape(1, -1)
        fs = scaler.transform(feat)
        if pca is not None:
            fs = pca.transform(fs)
        clus = int(kmeans.predict(fs)[0])
        cluster_counts[clus] = cluster_counts.get(clus, 0) + 1

        label = _label_from_cluster(clus, cluster_to_class, class_names)
        if label is None:
            unknown += 1
        else:
            counts[label] += 1

    # Persistir selección y conteos
    Path("muestra10_seleccion.txt").write_text("\n".join(sel), encoding="utf-8")
    Path("sample_counts.json").write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")

    # Guardar diagnóstico
    Path("diagnostico_clusters.json").write_text(
        json.dumps({"cluster_counts": cluster_counts, "unknown": unknown}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return sel, [counts.get(k, 0) for k in CLASSES_ORDER]

# ---------------- Bayes cajas ----------------
def posterior_over_boxes(counts):
    BOXES = {
        "a": [0.25, 0.25, 0.25, 0.25],
        "b": [0.15, 0.30, 0.30, 0.25],
        "c": [0.25, 0.35, 0.25, 0.15],
        "d": [0.50, 0.50, 0.00, 0.00],
    }
    PRIOR = {k: 1.0 / len(BOXES) for k in BOXES}
    EPS = 1e-9
    logs = {}
    for name, probs in BOXES.items():
        ll = 0.0
        for c, p in zip(counts, probs):
            p_eff = max(p, EPS)
            if c == 0:
                continue
            ll += c * np.log(p_eff)
        logs[name] = np.log(PRIOR[name]) + ll
    m = max(logs.values())
    exps = {k: np.exp(v - m) for k, v in logs.items()}
    Z = sum(exps.values())
    post = {k: v / Z for k, v in exps.items()}
    exp = [sum(post[b] * BOXES[b][i] for b in BOXES) for i in range(4)]
    pred1000 = [int(round(1000 * p)) for p in exp]
    return post, exp, pred1000

# ---------------- Voz (KNN) ----------------
def extract_features_audio(y, sr):
    target_len = int(0.8 * sr)
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(mat):
        return np.hstack([mat.mean(axis=1), mat.std(axis=1)])

    feat = np.hstack(
        [
            stats(mfcc),
            stats(d1),
            stats(d2),
            librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
            librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean(),
            librosa.feature.zero_crossing_rate(y).mean(),
        ]
    )
    return feat.astype(np.float32)

def predict_command_from_mic(clf):
    sg.popup_quick_message("Grabando...", auto_close_duration=1, background_color="yellow")
    rec = sd.rec(int(MIC_SECONDS * MIC_RATE), samplerate=MIC_RATE, channels=1, dtype="float32")
    sd.wait()
    y = rec.squeeze()
    feat = extract_features_audio(y, MIC_RATE).reshape(1, -1)
    return clf.predict(feat)[0]

def predict_command_from_wav(clf, wav_path):
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != MIC_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=MIC_RATE)
    feat = extract_features_audio(y, MIC_RATE).reshape(1, -1)
    return clf.predict(feat)[0]

def remap_clusters_ui(model_path):
    """
    Abre un dialogo para remapear cluster_id -> clase_canónica
    y guarda cluster_override.json.
    """
    try:
        bundle = load(model_path)
        kmeans = bundle["kmeans"]
        cluster_to_class = bundle.get("cluster_to_class", {})
        classes = bundle.get("classes", CLASSES_ORDER)

        # opciones seguras para dropdown
        choices = CLASSES_ORDER[:]  # ["tornillos","clavos","arandelas","tuercas"]

        # estado inicial: lo que venga del joblib, normalizado
        initial = {}
        for cid in range(kmeans.n_clusters):
            raw = cluster_to_class.get(cid, "")
            if isinstance(raw, int) and 0 <= raw < len(classes):
                raw = str(classes[raw])
            raw = str(raw).strip().lower()
            # normalizar por alias
            if raw in {"tornillo","tornillos"}: raw = "tornillos"
            elif raw in {"clavo","clavos"}: raw = "clavos"
            elif raw in {"arandela","arandelas"}: raw = "arandelas"
            elif raw in {"tuerca","tuercas"}: raw = "tuercas"
            else:
                raw = ""
            initial[cid] = raw

        # construir ventana sencilla con un dropdown por cluster
        layout = [[sg.Text("Remapear cluster → clase (se guarda en cluster_override.json)")],
                  [sg.HorizontalSeparator()]]
        rows = []
        for cid in range(kmeans.n_clusters):
            rows.append([sg.Text(f"Cluster {cid}:", size=(12,1)),
                         sg.Combo(values=choices, default_value=initial[cid] or choices[0],
                                  key=f"-C{cid}-", readonly=True, size=(16,1))])
        layout += rows
        layout += [[sg.HorizontalSeparator()],
                   [sg.Button("Guardar", key="-SAVE-"), sg.Button("Cancelar", key="-CANCEL-")]]

        win = sg.Window("Remap clusters", layout, modal=True, finalize=True)
        override = {}
        while True:
            ev, vals = win.read()
            if ev in (sg.WINDOW_CLOSED, "-CANCEL-"):
                win.close()
                return False
            if ev == "-SAVE-":
                for cid in range(kmeans.n_clusters):
                    lab = vals.get(f"-C{cid}-", "").strip().lower()
                    if lab not in choices:
                        win["-C0-"].Widget.bell()
                        sg.popup_error(f"Cluster {cid} tiene etiqueta inválida: {lab}")
                        break
                    override[cid] = lab
                else:
                    Path(OVERRIDE_PATH).write_text(json.dumps(override, indent=2, ensure_ascii=False), encoding="utf-8")
                    sg.popup_ok("Override guardado en cluster_override.json\nSe aplicará de inmediato.")
                    win.close()
                    return True
    except Exception as e:
        sg.popup_error(f"No se pudo remapear: {e}")
        return False


# ---------------- GUI ----------------
def main():
    global USE_CROP
    sg.theme("SystemDefault")

    layout = [
        [sg.Text("Agente IA — Demo Aleatoria + Subir imagen", font=("Segoe UI", 13, "bold"))],
        [sg.Text("K-Means:"), sg.Input(KMEANS_MODEL_PATH, key="-KMEANS-", size=(42, 1)), sg.FileBrowse("Buscar")],
        [sg.Text("Voz:"), sg.Input(VOICE_MODEL_PATH, key="-VOICE-", size=(42, 1)), sg.FileBrowse("Buscar")],
        [sg.Text("Dataset:"), sg.Input(DATASET_DIR, key="-DATASET-", size=(42, 1)), sg.FolderBrowse("Elegir")],
        [sg.Checkbox("Recorte por contorno (opcional)", key="-CROP-", default=False)],
        [sg.HorizontalSeparator()],
        [
            sg.Button("🎲 Elegir 10 al azar", key="-RUN-"),
            sg.Button("🎤 Escuchar (mic)", key="-MIC-"),
            sg.Button("🎧 Elegir WAV…", key="-WAV-"),
            sg.Button("📂 Subir imagen…", key="-UPLOAD-"),
            sg.Button("Salir", key="-EXIT-"),
            sg.Button("⚙️ Remap clusters", key="-REMAP-"),

        ],
        [sg.HorizontalSeparator()],
        [sg.Text("Salida:")],
        [sg.Multiline("", size=(100, 18), key="-OUT-", autoscroll=True, font=("Consolas", 10))],
    ]
    win = sg.Window("Agente IA - Desktop", layout, finalize=True)
    log = lambda msg: win["-OUT-"].print(msg)

    # Modelos de voz (lazy)
    bundle_v = None
    clf_voice = None

    while True:
        ev, vals = win.read()
        if ev in (sg.WINDOW_CLOSED, "-EXIT-"):
            break

        # Sincronizar flag de recorte con checkbox
        if ev in ("-RUN-", "-UPLOAD-"):
            USE_CROP = bool(vals.get("-CROP-", False))

        # Botón: Elegir 10 al azar
        if ev == "-RUN-":
            try:
                # Diagnóstico: ruta de modelo y mapeo
                kmeans_path = Path(vals["-KMEANS-"]).resolve()
                log(f"[DIAG] Usando modelo: {kmeans_path}")
                bundle = load(str(kmeans_path))
                log("[DIAG] classes: " + str(bundle.get("classes")))
                log("[DIAG] cluster_to_class: " + str(bundle.get("cluster_to_class")))

                sel, vec = classify_random_10(vals["-DATASET-"], str(kmeans_path))
                log("[VISIÓN] Muestra aleatoria elegida y clasificada.")
                log("[VISIÓN] Conteos (10): " + str(dict(zip(CLASSES_ORDER, vec))))
                log("Rutas guardadas en 'muestra10_seleccion.txt'.")

                # Diagnóstico de clusters de la muestra
                try:
                    diag = json.loads(Path("diagnostico_clusters.json").read_text(encoding="utf-8"))
                    log("[DIAG] dist de clusters en la muestra: " + str(diag.get("cluster_counts")))
                    if diag.get("unknown", 0) > 0:
                        log(f"[ALERTA] {diag['unknown']} imagen(es) cayeron en clusters sin mapeo/alias.")
                except Exception as e:
                    log(f"[DIAG] No se pudo leer diagnóstico: {e}")

            except Exception as e:
                log(f"[ERROR VISIÓN] {e}")

        # Botón: Escuchar mic
        if ev == "-MIC-":
            try:
                if clf_voice is None:
                    voice_path = Path(vals["-VOICE-"]).resolve()
                    bundle_v = load(str(voice_path))
                    clf_voice = bundle_v["pipeline"]
                    log(f"✅ Modelo de voz cargado: {voice_path.name}")
                pred = predict_command_from_mic(clf_voice)
                log(f"[VOZ] Comando: {pred}")

                # Acciones
                cmd = str(pred).lower()
                if cmd.startswith("cont"):
                    # Recuperar conteos si existen
                    if Path("sample_counts.json").exists():
                        data = json.loads(Path("sample_counts.json").read_text(encoding="utf-8"))
                        sample_counts = [int(data.get(k, 0)) for k in CLASSES_ORDER]
                        log(f"[ACCION] Conteos → {dict(zip(CLASSES_ORDER, sample_counts))}")
                    else:
                        log("No hay conteos. Usá 'Elegir 10 al azar'.")
                elif cmd.startswith("prop"):
                    if Path("sample_counts.json").exists():
                        data = json.loads(Path("sample_counts.json").read_text(encoding="utf-8"))
                        sample_counts = [int(data.get(k, 0)) for k in CLASSES_ORDER]
                        post, exp, pred1000 = posterior_over_boxes(sample_counts)
                        log("[ACCION] Proporciones bayesianas:")
                        for name, p, n in zip(CLASSES_ORDER, exp, pred1000):
                            log(f"  {name:<10} {p*100:6.2f}%  ({n}/1000)")
                        log(f"  Caja MAP: {max(post, key=post.get)}")
                    else:
                        log("No hay conteos. Usá 'Elegir 10 al azar'.")
                elif cmd.startswith("sal"):
                    log("[ACCION] Finalizar")
                    break
                else:
                    log("Comando no reconocido.")
            except Exception as e:
                log(f"[ERROR VOZ] {e}")

        # Botón: Elegir WAV
        if ev == "-WAV-":
            path = sg.popup_get_file("Elegí un WAV", file_types=(("WAV", "*.wav"),))
            if path:
                try:
                    if clf_voice is None:
                        voice_path = Path(vals["-VOICE-"]).resolve()
                        bundle_v = load(str(voice_path))
                        clf_voice = bundle_v["pipeline"]
                        log(f"✅ Modelo de voz cargado: {voice_path.name}")
                    pred = predict_command_from_wav(clf_voice, path)
                    log(f"[VOZ] Comando (wav): {pred}")
                except Exception as e:
                    log(f"[ERROR VOZ] {e}")

        # 📂 Botón: Subir imagen…
        if ev == "-UPLOAD-":
            try:
                img_path = sg.popup_get_file(
                    "Elegí una imagen",
                    file_types=(("Imagen", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp"),),
                )
                if img_path:
                    kmeans_path = Path(vals["-KMEANS-"]).resolve()
                    label, clus = classify_one_image(img_path, str(kmeans_path))
                    if label is None:
                        log(f"[UPLOAD] {Path(img_path).name} → desconocido (cluster {clus})")
                    else:
                        log(f"[UPLOAD] {Path(img_path).name} → {label} (cluster {clus})")
            except Exception as e:
                log(f"[ERROR UPLOAD] {e}")

        if ev == "-REMAP-":
            try:
                kmeans_path = Path(vals["-KMEANS-"]).resolve()
                if not kmeans_path.exists():
                    sg.popup_error("Seleccioná primero el archivo .joblib de K-Means.")
                else:
                    ok = remap_clusters_ui(str(kmeans_path))
                    if ok:
                        # mostrar cómo quedó
                        override = json.loads(Path(OVERRIDE_PATH).read_text(encoding="utf-8"))
                        log("[REMAP] Nuevo cluster_override.json: " + str(override))
            except Exception as e:
                log(f"[ERROR REMAP] {e}")


    win.close()

if __name__ == "__main__":
    main()
