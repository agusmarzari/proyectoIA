#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clasificación de UNA imagen usando el bundle entrenado:
- Carga kmeans_bundle.joblib (scaler, pca, kmeans)
- Extrae features (Hu + LBP + HSV) exactamente igual
- Predice cluster y, si existe, muestra etiqueta de cluster_to_class.json
- Guarda una máscara debug_* para verificar segmentación
Uso:
  python clasificar_una.py ruta/a/imagen.jpg
"""

import os, sys, json
import numpy as np
import cv2
from joblib import load
from skimage.feature import local_binary_pattern

# --- mismos parámetros que el entrenamiento ---
IMG_SIZE = (256, 256)
LBP_P, LBP_R, LBP_METHOD = 8, 1, "uniform"
HSV_BINS = 16

BUNDLE  = "kmeans_bundle.joblib"
MAPJSON = "cluster_to_class.json"


def preprocess_mask(gray):
    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = th.astype(np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return th
    mask = np.zeros_like(th)
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(mask, [c], -1, 1, thickness=-1)
    return mask


def hu_moments_from_mask(mask):
    m = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu.astype(np.float32)


def lbp_hist(gray):
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2
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
        raise SystemExit(f"No puedo leer {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = preprocess_mask(gray)

    hu  = hu_moments_from_mask(mask)
    lbp = lbp_hist(gray)
    hsv = hsv_hist(img)
    feat = np.hstack([hu, lbp, hsv]).astype(np.float32)

    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(f"debug_mask_{base}.png", (mask*255).astype("uint8"))
    return feat


def main():
    if len(sys.argv) < 2:
        print("Uso: python clasificar_una.py ruta/a/imagen.jpg")
        sys.exit(1)
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        raise SystemExit(f"No existe: {img_path}")

    if not os.path.exists(BUNDLE):
        raise SystemExit("No encuentro kmeans_bundle.joblib. Corré primero kmeans_mejorado.py")

    bundle = load(BUNDLE)
    scaler = bundle["scaler"]
    pca    = bundle["pca"]
    kmeans = bundle["kmeans"]

    x  = extract_features_img(img_path).reshape(1, -1)
    xs = scaler.transform(x)
    xf = pca.transform(xs) if pca is not None else xs

    lab = int(kmeans.predict(xf)[0])
    dists = kmeans.transform(xf)[0]

    # nombre de clase, si existe mapping
    name = f"cluster_{lab}"
    if os.path.exists(MAPJSON):
        with open(MAPJSON, "r", encoding="utf-8") as f:
            m = json.load(f)
        # lookup robusto (admite "2" o 2 como clave)
        name = m.get(str(lab), m.get(lab, name))

    print("\n== Clasificación ==")
    print("Imagen:", img_path)
    print("Cluster predicho:", lab)
    print("Etiqueta:", name)
    print("Distancias a centroides (menor = mejor):")
    # si el mapping tiene 4 entradas, las muestro en orden 0..3
    for i, d in enumerate(dists):
        print(f"  {i}: {d:.6f}")
    print("\nSe guardó la máscara como debug_mask_*.png (verificá segmentación).")


if __name__ == "__main__":
    main()
