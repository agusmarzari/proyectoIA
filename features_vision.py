#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from PIL import Image
import pillow_heif  # pip install pillow pillow-heif

# --- Config del extractor "de entrenamiento" (el que te funcionó bien) ---
LBP_P, LBP_R = 8, 1
LBP_METHOD = "uniform"
HSV_BINS = (8, 8, 8)  # 512 bins

def _read_image_any(data_or_mat):
    """
    Acepta bytes (upload) o np.ndarray (RGB/BGR). Devuelve BGR o None.
    """
    img = None
    if isinstance(data_or_mat, (bytes, bytearray)):
        arr = np.frombuffer(data_or_mat, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            # HEIC -> PIL -> BGR
            try:
                heif = pillow_heif.read_heif(data_or_mat)
                pil = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                img = None
    else:
        mat = data_or_mat
        if mat is not None:
            if mat.ndim == 3 and mat.shape[2] == 3:
                # asumimos RGB y convertimos a BGR
                img = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            else:
                img = mat
    return img

def _preprocess_mask_training(gray):
    """
    Mismo preprocesado que usamos al armar los clusters:
    equalizeHist -> GaussianBlur -> AdaptiveThreshold (GAUSSIAN) -> BIN_INV
    """
    g = cv2.equalizeHist(gray)
    g = cv2.GaussianBlur(g, (5,5), 0)
    bw = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    return bw

def _hu_from_mask(mask):
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu

def _lbp_hist(gray):
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2  # 10 bins para P=8 uniform
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def _hsv_hist(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, HSV_BINS, [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features_image_training(data_or_mat):
    """
    Devuelve:
      - feat: np.ndarray shape (529,), dtype float32
      - img_out: imagen BGR redimensionada (256x256) para preview (o None)
    """
    img = _read_image_any(data_or_mat)
    if img is None:
        return None, None

    img = cv2.resize(img, (256,256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = _preprocess_mask_training(gray)
    hu  = _hu_from_mask(mask)                 # 7
    lbp = _lbp_hist(gray)                     # 10
    hsv = _hsv_hist(img)                      # 512
    feat = np.hstack([hu, lbp, hsv]).astype(np.float32)  # 7+10+512 = 529

    return feat, img
