#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump

CLASSES = ["tornillos","clavos","arandelas","tuercas"]
LBP_P, LBP_R = 8, 1
LBP_METHOD = "uniform"
HSV_BINS = (8, 8, 8)
SEED = 42

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

def extract_features(path):
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

def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)

def build_from_dataset(dataset_dir, use_pca=True, pca_dim=50, protos_dir=None, out_path="kmeans_piezas.joblib"):
    print(f"[INFO] Dataset: {dataset_dir}")
    paths = list_images(dataset_dir)
    if len(paths) == 0:
        raise SystemExit("No se encontraron imágenes en el dataset.")

    # Extract features
    X = []
    for p in paths:
        try:
            X.append(extract_features(p))
        except Exception as e:
            print("WARN:", p, e)
    X = np.asarray(X)
    print(f"[INFO] Imágenes válidas: {len(X)}  |  Dim feats: {X.shape[1]}")

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA (optional)
    pca = None
    if use_pca:
        pca_dim = int(min(pca_dim, Xs.shape[0]-1, Xs.shape[1]))
        print(f"[INFO] PCA activado → dim={pca_dim}")
        pca = PCA(n_components=pca_dim, random_state=SEED)
        Xf = pca.fit_transform(Xs)
    else:
        Xf = Xs

    # Init KMeans
    if protos_dir and Path(protos_dir).exists():
        init_centers = []
        proto_map = {}
        for cname in CLASSES:
            folder = Path(protos_dir)/cname
            imgs = list_images(str(folder))
            if len(imgs) == 0:
                raise SystemExit(f"No hay prototipos para {cname} en {folder}")
            feats = [extract_features(p) for p in imgs]
            feats = scaler.transform(np.asarray(feats))
            if pca is not None:
                feats = pca.transform(feats)
            center = feats.mean(axis=0)
            init_centers.append(center)
            proto_map[len(init_centers)-1] = cname
        init_centers = np.asarray(init_centers)
        print("[INFO] Inicializando KMeans con centros de prototipos…")
        kmeans = KMeans(n_clusters=4, init=init_centers, n_init=1, random_state=SEED)
        kmeans.fit(Xf)
        cluster_to_class = {int(i): proto_map[i] for i in range(4)}  # naming directo por prototipo
    else:
        print("[INFO] Inicializando KMeans con k-means++ (sin prototipos)…")
        kmeans = KMeans(n_clusters=4, init="k-means++", n_init=20, random_state=SEED)
        kmeans.fit(Xf)
        cluster_to_class = {int(i): None for i in range(4)}  # deberás editar luego

    bundle = {
        "kmeans": kmeans,
        "scaler": scaler,
        "pca": pca,
        "classes": CLASSES,
        "cluster_to_class": cluster_to_class
    }
    dump(bundle, out_path)
    print(f"[OK] Modelo guardado en {out_path}")
    if all(v is None for v in cluster_to_class.values()):
        map_path = Path(out_path).with_suffix(".mapping.json")
        map_example = {str(i): "tornillos/clavos/arandelas/tuercas" for i in range(4)}
        map_path.write_text(json.dumps(map_example, indent=2), encoding="utf-8")
        print(f"[ATENCIÓN] No se especificaron prototipos. Se creó {map_path} como ejemplo para que asignes manualmente cada cluster a una clase.")
    return out_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Carpeta con imágenes (p.ej. dataset_mix)")
    ap.add_argument("--protos", default=None, help="Carpeta con prototipos por clase (protos/{tornillos,clavos,arandelas,tuercas})")
    ap.add_argument("--out", default="kmeans_piezas.joblib")
    ap.add_argument("--use_pca", type=int, default=1)
    ap.add_argument("--pca_dim", type=int, default=50)
    args = ap.parse_args()

    build_from_dataset(args.dataset, use_pca=bool(args.use_pca), pca_dim=args.pca_dim, protos_dir=args.protos, out_path=args.out)
