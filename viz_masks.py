#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys
from pathlib import Path
import cv2
import numpy as np

EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")

def imread(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No se pudo leer {path}")
    return img

def to_panel(imgs, cols=2):
    # arma un collage simple (mismo tamaño)
    h, w = imgs[0].shape[:2]
    rows = int(np.ceil(len(imgs)/cols))
    canvas = np.ones((rows*h, cols*w, 3), dtype=np.uint8)*255
    for i,im in enumerate(imgs):
        r, c = i//cols, i%cols
        if im.ndim==2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im
    return canvas

def make_mask(img_bgr, method="hsv_adapt", blur_ks=5, morph_open=1, morph_close=1):
    """
    method:
      - 'hsv_adapt' : adaptiveThreshold sobre canal V ecualizado (robusto)
      - 'gray_otsu' : Otsu sobre gris (suave)
      - 'gray_adapt': adaptiveThreshold sobre gris (como tu pipeline clásico)
    """
    if blur_ks < 1: blur_ks = 1
    if method == "hsv_adapt":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        V   = hsv[...,2]
        V   = cv2.equalizeHist(V)
        Vb  = cv2.GaussianBlur(V, (blur_ks|1, blur_ks|1), 0)
        bw  = cv2.adaptiveThreshold(Vb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 5)
    elif method == "gray_otsu":
        g   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g   = cv2.equalizeHist(g)
        g   = cv2.GaussianBlur(g, (blur_ks|1, blur_ks|1), 0)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "gray_adapt":
        g   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g   = cv2.equalizeHist(g)
        g   = cv2.GaussianBlur(g, (blur_ks|1, blur_ks|1), 0)
        bw  = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 5)
    else:
        raise ValueError(f"Método desconocido: {method}")

    # morfología (limpieza)
    if morph_open > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=morph_open)
    if morph_close > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=morph_close)
    return bw

def overlay_contour(img_bgr, mask_bw, color=(0,255,0)):
    out = img_bgr.copy()
    cnts,_ = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out

def main():
    ap = argparse.ArgumentParser(description="Visualizar máscaras de dataset (debug).")
    ap.add_argument("--in_dir",  default="dataset_mix", help="Carpeta con imágenes mezcladas")
    ap.add_argument("--out_dir", default="masks_debug", help="Salida para máscaras y paneles")
    ap.add_argument("--method",  default="hsv_adapt",
                    choices=["hsv_adapt","gray_otsu","gray_adapt"],
                    help="Estrategia de umbral")
    ap.add_argument("--resize",  type=int, default=512,
                    help="Lado a reescalar para el debug (0 = no redimensionar)")
    ap.add_argument("--open",    type=int, default=1, help="Iteraciones MORPH_OPEN (limpia ruido)")
    ap.add_argument("--close",   type=int, default=1, help="Iteraciones MORPH_CLOSE (cierra huecos)")
    ap.add_argument("--limit",   type=int, default=0, help="Procesar solo N primeras (0 = todas)")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_masks   = out_dir/"masks"
    out_overlay = out_dir/"overlay"
    out_panels  = out_dir/"panels"

    for p in (out_dir, out_masks, out_overlay, out_panels):
        p.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in in_dir.iterdir() if p.suffix.lower() in EXTS]
    if not imgs:
        print(f"[ERROR] No encontré imágenes en {in_dir}", file=sys.stderr); sys.exit(1)
    if args.limit > 0:
        imgs = imgs[:args.limit]

    print(f"[INFO] Imágenes: {len(imgs)}  | método: {args.method}")
    for i, path in enumerate(imgs, 1):
        try:
            img = imread(path)
            if args.resize and args.resize > 0:
                img = cv2.resize(img, (args.resize, args.resize))

            mask = make_mask(img, method=args.method,
                             morph_open=args.open, morph_close=args.close)
            ov   = overlay_contour(img, mask)
            # panel 2x2: original, máscara, overlay, máscara invertida
            panel = to_panel([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                              ov, cv2.cvtColor(255-mask, cv2.COLOR_GRAY2BGR)], cols=2)

            stem = path.stem
            cv2.imwrite(str(out_masks / f"{stem}_mask.png"), mask)
            cv2.imwrite(str(out_overlay / f"{stem}_overlay.jpg"), ov)
            cv2.imwrite(str(out_panels / f"{stem}_panel.jpg"), panel)

            if i % 50 == 0 or i == len(imgs):
                print(f"[{i}/{len(imgs)}] {path.name}")
        except Exception as e:
            print(f"[WARN] {path.name}: {e}")

    print(f"[OK] Guardado en: {out_dir}\n  - masks/\n  - overlay/\n  - panels/")

if __name__ == "__main__":
    main()
