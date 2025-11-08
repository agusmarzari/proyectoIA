#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, glob, json
import numpy as np
import cv2
from joblib import load, dump
from skimage.feature import local_binary_pattern

CLASSES = ["tornillos","clavos","arandelas","tuercas"]
LBP_P, LBP_R = 8, 1
LBP_METHOD = "uniform"
HSV_BINS = (8,8,8)

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
    p = Path(folder)
    for e in exts:
        paths.extend([str(x) for x in p.glob(e)])
    return sorted(paths)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="kmeans_piezas.joblib")
    ap.add_argument("--protos", required=True, help="carpeta con subcarpetas nombradas por clase real")
    args = ap.parse_args()

    bundle = load(args.model)
    kmeans = bundle["kmeans"]
    scaler = bundle["scaler"]
    pca = bundle.get("pca", None)

    protos_dir = Path(args.protos)
    classes_found = [d.name for d in protos_dir.iterdir() if d.is_dir()]
    if not classes_found:
        raise SystemExit("No se encontraron subcarpetas dentro de --protos")
    print("[INFO] Clases detectadas en protos:", classes_found)

    target_classes = CLASSES if all((protos_dir/c).exists() for c in CLASSES) else classes_found

    votes = {}
    for cname in target_classes:
        folder = protos_dir / cname
        imgs = list_images(folder)
        if len(imgs) == 0:
            print(f"[WARN] No hay imágenes en {folder}, salto")
            continue
        for p in imgs:
            feat = extract_features(p).reshape(1,-1)
            fs = scaler.transform(feat)
            if pca is not None:
                fs = pca.transform(fs)
            clus = int(kmeans.predict(fs)[0])
            votes.setdefault(clus, {}).setdefault(cname, 0)
            votes[clus][cname] += 1

    mapping = {}
    for clus, tally in votes.items():
        best_class = max(tally.items(), key=lambda kv: kv[1])[0]
        mapping[clus] = best_class

    print("[INFO] Votos por cluster:", votes)
    print("[INFO] Mapeo propuesto cluster→clase:", mapping)
    bundle["cluster_to_class"] = mapping
    dump(bundle, args.model)
    map_json = Path(args.model).with_suffix(".mapping.json")
    map_json.write_text(json.dumps({int(k): v for k,v in mapping.items()}, indent=2), encoding="utf-8")
    print(f"[OK] Actualizado {args.model} con mapping. También guardado {map_json}")

if __name__ == "__main__":
    main()
