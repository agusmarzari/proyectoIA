#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means mejorado para agrupar imágenes mezcladas en 4 clusters.

Mejoras clave:
- Features combinadas: Forma (Hu), Textura (LBP), Color (HSV).
- Normalización con StandardScaler.
- Opción de PCA (dimensión reducida para compactar/ruido).
- KMeans con k-means++ y varios reinicios (n_init).
- Salida: carpetas por cluster + CSV con asignaciones.
- (Opcional) Inicialización semi-supervisada con prototipos por clase.

Uso básico (sin prototipos):
    1) Coloca tus imágenes en "dataset_mix/".
    2) Ejecuta:
        python kmeans_mejorado.py
    3) Mira resultados en "out_clusters_mejorado/" y "cluster_assignments.csv".

Uso opcional con prototipos (si quieres forzar mejor separación):
    1) Crea "protos/" con subcarpetas por clase y 1-3 imágenes representativas:
        protos/
          tornillos/
          tuercas/
          arandelas/
          clavos/
    2) Activa USE_PROTOTYPES = True abajo.
    3) Ejecuta el script. Tomará esos prototipos como centros iniciales.
Requisitos:
    pip install opencv-python scikit-image scikit-learn numpy joblib pandas
"""
import os, glob, shutil, csv
import numpy as np
import cv2
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------
# CONFIG
# -----------------------
INPUT_DIR = "dataset_mix"                  # carpeta mezclada
OUTPUT_DIR = "out_clusters_mejorado"       # salida
CSV_OUT = "cluster_assignments.csv"
K = 4
SEED = 42
N_INIT = 20                                # más robusto que 'auto' para datasets chicos
USE_PCA = False
PCA_DIM = 10                               # compresión (ajústalo si quieres)
# Prototipos (opcional)
USE_PROTOTYPES = False
PROTOS_DIR = "protos"
PROTO_CLASSES = ["tornillos", "tuercas", "arandelas", "clavos"]

# LBP
LBP_P, LBP_R = 8, 1
LBP_METHOD = "uniform"                     # hist bins = P + 2
# HSV hist
HSV_BINS = (8, 8, 8)                       # 512 dims

# -----------------------
# Utilidades
# -----------------------
def ensure_dirs(base: str, k: int):
    os.makedirs(base, exist_ok=True)
    for i in range(k):
        os.makedirs(os.path.join(base, f"cluster_{i}"), exist_ok=True)

def list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)

def preprocess_mask(gray):
    # Mejorar contraste y resaltar figura principal
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )
    return bw

def hu_moments_from_mask(mask):
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    # estabilizar escala log
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu  # 7 dims

def lbp_hist(gray, P=LBP_P, R=LBP_R):
    lbp = local_binary_pattern(gray, P, R, method=LBP_METHOD)
    n_bins = P + 2  # "uniform"
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist  # ~10 dims

def hsv_hist(img_bgr, bins=HSV_BINS):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist  # 512 dims

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo leer {img_path}")
    img = cv2.resize(img, (256,256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = preprocess_mask(gray)
    hu = hu_moments_from_mask(mask)    # 7
    lbp = lbp_hist(gray)               # ~10
    hsv = hsv_hist(img)                # 512
    return np.hstack([hu, lbp, hsv])   # total ~529

def extract_matrix(paths):
    X, ok_paths = [], []
    for p in paths:
        try:
            feat = extract_features(p)
            X.append(feat)
            ok_paths.append(p)
        except Exception as e:
            print("WARN:", p, e)
    if len(X) == 0:
        raise SystemExit("No se pudieron extraer features de ninguna imagen.")
    return np.array(X), ok_paths

def compute_prototype_centers(scaler, pca=None):
    """Devuelve centros iniciales a partir de imágenes prototipo por clase."""
    centers = []
    for cname in PROTO_CLASSES:
        folder = os.path.join(PROTOS_DIR, cname)
        imgs = list_images(folder)
        if len(imgs) == 0:
            raise SystemExit(f"Faltan prototipos en {folder}")
        feats, _ = extract_matrix(imgs)
        feats = scaler.transform(feats)
        if pca is not None:
            feats = pca.transform(feats)
        center = feats.mean(axis=0)
        centers.append(center)
    return np.vstack(centers)

# -----------------------
# Main
# -----------------------
def main():
    # 1) Leer imágenes
    paths = list_images(INPUT_DIR)
    if len(paths) == 0:
        raise SystemExit("No se encontraron imágenes en dataset_mix/. Coloca tus imágenes y vuelve a ejecutar.")
    print(f"Imágenes encontradas: {len(paths)}")

    # 2) Extraer features
    X, ok_paths = extract_matrix(paths)
    print(f"Imágenes válidas: {len(ok_paths)}  |  Dim feats: {X.shape[1]}")

    # 3) Escalar
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 4) PCA opcional
    if USE_PCA:
        pca = PCA(n_components=min(PCA_DIM, Xs.shape[1]), random_state=SEED)
        Xf = pca.fit_transform(Xs)
        print(f"PCA activado → nueva dimensión: {Xf.shape[1]}")
    else:
        pca = None
        Xf = Xs

    # 5) K-Means (con o sin prototipos)
    if USE_PROTOTYPES:
        # Centros a partir de protos por clase
        init_centers = compute_prototype_centers(scaler, pca)
        if init_centers.shape[0] != K:
            raise SystemExit("La cantidad de prototipos no coincide con K.")
        print("Inicializando KMeans con centros de prototipos…")
        kmeans = KMeans(n_clusters=K, init=init_centers, n_init=1, random_state=SEED)
    else:
        print(f"Inicializando KMeans k-means++ | n_init={N_INIT}…")
        kmeans = KMeans(n_clusters=K, init="k-means++", n_init=N_INIT, random_state=SEED)

    labels = kmeans.fit_predict(Xf)

    # 6) Exportar resultados: carpetas + CSV
    ensure_dirs(OUTPUT_DIR, K)
    counts = {i: 0 for i in range(K)}
    rows = []
    for p, lab in zip(ok_paths, labels):
        dst = os.path.join(OUTPUT_DIR, f"cluster_{int(lab)}", os.path.basename(p))
        shutil.copy2(p, dst)
        counts[int(lab)] += 1
        rows.append({"filename": os.path.basename(p), "path": p, "cluster": int(lab)})

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","path","cluster"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nDistribución por cluster:")
    for i in range(K):
        print(f"  cluster_{i}: {counts[i]} imágenes")

    print(f"\nListo. Revisa: {OUTPUT_DIR}  y  {CSV_OUT}")
    if USE_PROTOTYPES:
        print("TIP: puedes ajustar tus prototipos para mejorar separación.")

if __name__ == "__main__":
    main()
