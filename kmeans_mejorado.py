#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means mejorado para piezas (tuercas, tornillos, arandelas, clavos)
- Lee imágenes de dataset_mix/
- Extrae features (Hu + LBP + HSV) sobre máscara binaria (fondo claro)
- Escala con StandardScaler, (opcional) PCA
- Entrena KMeans (k=4), exporta asignaciones a out_clusters_mejorado/cluster_*
- Guarda bundle para inferencia: kmeans_bundle.joblib  (scaler, pca, kmeans)
- Guarda mapping cluster->nombre de carpeta: cluster_to_class.json
"""

import os, glob, csv, re, json, shutil
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Si querés usar PCA, destapá las dos líneas marcadas con "PCA"
# from sklearn.decomposition import PCA

from skimage.feature import local_binary_pattern

# ---------------- Config ----------------
DATASET_DIR = "dataset_mix"
OUT_DIR     = "out_clusters_mejorado"
ASSIGN_CSV  = "cluster_assignments.csv"
BUNDLE_OUT  = "kmeans_bundle.joblib"
MAPJSON     = "cluster_to_class.json"
IMG_SIZE    = (256, 256)

# LBP (uniform)
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"

# HSV hist bins por canal
HSV_BINS = 16  # 16x3 = 48 dims

# KMeans
N_CLUSTERS = 4
N_INIT     = 20
RSTATE     = 42
# ----------------------------------------


# --------------- Feature helpers ----------------
def preprocess_mask(gray):
    """Fondo claro: genera máscara binaria del objeto, preservando agujeros internos
    (tuercas / arandelas)."""
    # 1) Normalizar contraste + suavizar
    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # 2) Umbral inverso (objeto más oscuro/medio sobre fondo claro)
    #    -> output en {0,1} para operar como máscara lógica
    _, th = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = th.astype(np.uint8)

    # 3) Morfología suave: abrir para quitar puntitos.
    #    (Evitar cierre fuerte que taparía el agujero de la tuerca)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # 4) Encontrar contornos preservando jerarquía (externos e internos)
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Si no hay contornos, devolver tal cual
    if hier is None or len(cnts) == 0:
        return th

    hier = hier[0]  # jerarquía: [Next, Prev, FirstChild, Parent]
    mask = np.zeros_like(th, dtype=np.uint8)

    # 5) Rellenar contornos externos con 1 y "recortar" los internos con 0
    #    (así los agujeros quedan negros dentro del objeto blanco)
    for i, cnt in enumerate(cnts):
        parent = hier[i][3]
        if parent == -1:
            # contorno externo
            cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
        else:
            # agujero interno
            cv2.drawContours(mask, [cnt], -1, 0, thickness=-1)

    # 6) (Opcional) un cierre MUY suave solo para bordes dentados, sin tapar agujeros finos
    #    Si ves que deja “dientes” en el borde, podés activar esta línea:
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    return mask


def hu_moments_from_mask(mask):
    m = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(m).flatten()
    # log transform para estabilizar
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
        hist = cv2.calcHist([hsv],[ch],None,[HSV_BINS],[0, 256])
        hist = hist.flatten().astype(np.float32)
        hist = hist / (hist.sum() + 1e-12)
        hs.append(hist)
    return np.hstack(hs).astype(np.float32)


def extract_features_img(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No puedo leer: {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = preprocess_mask(gray)

    hu  = hu_moments_from_mask(mask)          # 7
    lbp = lbp_hist(gray)                      # 10 (uniform con P=8)
    hsv = hsv_hist(img)                       # 48 (16*3)
    feat = np.hstack([hu, lbp, hsv]).astype(np.float32)  # total 65 dims
    return feat, mask
# -------------------------------------------------


def load_dataset_images():
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp","*.JPG","*.JPEG","*.PNG")
    seen = set()
    files = []
    base = os.path.abspath(DATASET_DIR)
    for e in exts:
        for fp in glob.glob(os.path.join(base, e)):
            # clave única por ruta absoluta normalizada (insensible a mayúsculas en Windows)
            key = os.path.normcase(os.path.abspath(fp))
            if key in seen:
                continue
            seen.add(key)
            files.append(fp)
    return sorted(files)



def main():
    files = load_dataset_images()
    print(f"Imágenes encontradas: {len(files)}")
    feats, good_files, masks = [], [], {}

    for fp in files:
        try:
            f, m = extract_features_img(fp)
            feats.append(f)
            good_files.append(fp)
            # guardo máscaras por si querés revisar
            base = os.path.splitext(os.path.basename(fp))[0]
            masks[fp] = (m*255).astype("uint8")
        except Exception as e:
            print(f"[AVISO] Salteo {fp}: {e}")

    if not feats:
        raise SystemExit("No hay imágenes válidas en dataset_mix/")

    X = np.vstack(feats)
    print(f"Imágenes válidas: {X.shape[0]}  |  Dim feats: {X.shape[1]}")

    # ========== Escalado (y PCA opcional) ==========
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Si querés PCA, destapá este bloque:
    # pca = PCA(n_components=min(50, X_scaled.shape[1]))
    # X_model = pca.fit_transform(X_scaled)
    pca = None
    X_model = X_scaled
    # ===============================================

    print(f"Inicializando KMeans k-means++ | n_init={N_INIT}")
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=RSTATE)
    kmeans.fit(X_model)
    labels = kmeans.labels_.astype(int)

    # ---- volcar clusters a carpetas ----
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, fp in enumerate(good_files):
        lab = int(labels[i])
        outc = os.path.join(OUT_DIR, f"cluster_{lab}")
        os.makedirs(outc, exist_ok=True)
        dst = os.path.join(outc, os.path.basename(fp))
        if os.path.abspath(fp) != os.path.abspath(dst):
            shutil.copy2(fp, dst)
        # máscara debug opcional por si querés verla
        # cv2.imwrite(os.path.join(outc, f"mask_{os.path.basename(fp)}.png"), masks[fp])

    # ---- CSV de asignaciones ----
    with open(ASSIGN_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "cluster"])
        for i, fp in enumerate(good_files):
            wr.writerow([os.path.basename(fp), int(labels[i])])

    # ---- resumen ----
    counts = {f"cluster_{k}": int(np.sum(labels==k)) for k in range(N_CLUSTERS)}
    print("\nDistribución por cluster:")
    for name in sorted(counts):
        print(f"  {name}: {counts[name]} imágenes")

    print("\nListo. Revisa: out_clusters_mejorado  y  cluster_assignments.csv")

    # ---------- Guardar bundle ----------
    try:
        from joblib import dump
        bundle = {
            "scaler": scaler,
            "pca": pca,          # None si no usaste PCA
            "kmeans": kmeans,
            "feature_cfg": {
                "IMG_SIZE": IMG_SIZE,
                "LBP_P": LBP_P, "LBP_R": LBP_R, "LBP_METHOD": LBP_METHOD,
                "HSV_BINS": HSV_BINS
            }
        }
        dump(bundle, BUNDLE_OUT)
        print(f"\n[OK] Bundle guardado en {BUNDLE_OUT}")

        # mapping cluster -> nombre de carpeta (cluster_0, etc.)
        mapping = {}
        if os.path.isdir(OUT_DIR):
            for name in sorted(os.listdir(OUT_DIR)):
                p = os.path.join(OUT_DIR, name)
                if not os.path.isdir(p): 
                    continue
                m = re.search(r"(\d+)$", name)  # lee el número final
                if m:
                    mapping[int(m.group(1))] = name
            with open(MAPJSON, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            print(f"[OK] Mapping guardado en {MAPJSON}: {mapping}")
        else:
            print("[AVISO] No encontré out_clusters_mejorado; sin mapping.")

    except Exception as e:
        print("[ERROR] No se pudo generar el bundle:", e)


if __name__ == "__main__":
    main()
