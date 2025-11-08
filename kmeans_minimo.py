#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means mínimo para agrupar imágenes mezcladas en 4 clusters.
Uso:
    1) Coloca tus imágenes dentro de una carpeta llamada "dataset_mix/".
    2) Ejecuta:
        python kmeans_minimo.py
    3) Se crearán subcarpetas en "out_clusters/cluster_0..3" con las imágenes agrupadas.
Requisitos:
    pip install opencv-python scikit-learn numpy
"""
import os, glob, shutil
import numpy as np
import cv2
from sklearn.cluster import KMeans

# ---- CONFIG ----
INPUT_DIR = "dataset_mix"        # tu única carpeta mezclada
OUTPUT_DIR = "out_clusters"      # se creará con subcarpetas por cluster
K = 4                            # 4 tipos de piezas
HSV_BINS = (8, 8, 8)             # histograma compacto
SEED = 42

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

def hsv_hist(img_bgr, bins=HSV_BINS):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def main():
    ensure_dirs(OUTPUT_DIR, K)
    paths = list_images(INPUT_DIR)
    if len(paths) == 0:
        raise SystemExit("No se encontraron imágenes en dataset_mix/. Coloca tus imágenes y vuelve a ejecutar.")
    
    X, ok_paths = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print("WARN: no se pudo leer:", p)
            continue
        img = cv2.resize(img, (256,256))
        feat = hsv_hist(img)
        X.append(feat)
        ok_paths.append(p)
    
    if len(X) == 0:
        raise SystemExit("No se pudieron cargar imágenes válidas.")
    
    X = np.array(X)
    print(f"Imágenes válidas: {len(X)}")
    
    kmeans = KMeans(n_clusters=K, random_state=SEED, n_init="auto")
    labels = kmeans.fit_predict(X)
    
    counts = {i: 0 for i in range(K)}
    for p, lab in zip(ok_paths, labels):
        dst = os.path.join(OUTPUT_DIR, f"cluster_{int(lab)}", os.path.basename(p))
        shutil.copy2(p, dst)
        counts[int(lab)] += 1
    
    print("\nDistribución por cluster:")
    for i in range(K):
        print(f"  cluster_{i}: {counts[i]} imágenes")
    print(f"\nListo. Revisa las carpetas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
