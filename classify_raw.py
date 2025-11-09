import sys, json
import numpy as np
import cv2
from pathlib import Path
from joblib import load
from skimage.feature import local_binary_pattern

MODEL = "kmeans_piezas.joblib"  # cambia si hace falta

LBP_P, LBP_R, LBP_METHOD = 8, 1, "uniform"
HSV_BINS = (8,8,8)

def extract_features_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo leer {path}")
    img = cv2.resize(img, (256,256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 5)
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

def main():
    if len(sys.argv) < 2:
        print("Uso: python classify_raw.py img1.jpg [img2.jpg ...]")
        sys.exit(1)

    b = load(MODEL)
    kmeans = b["kmeans"]
    scaler = b["scaler"]
    pca = b.get("pca", None)

    print("== Model path:", Path(MODEL).resolve())
    print("n_clusters:", kmeans.n_clusters)

    for p in sys.argv[1:]:
        x = extract_features_image(p).reshape(1,-1)
        xs = scaler.transform(x)
        if pca is not None:
            xs = pca.transform(xs)
        # pred cluster crudo
        clus = int(kmeans.predict(xs)[0])
        # distancia a todos los centroides
        ctrs = kmeans.cluster_centers_
        dists = np.linalg.norm(xs - ctrs, axis=1)
        order = np.argsort(dists)
        print(f"\nImagen: {p}")
        print("  cluster_pred:", clus)
        print("  distancias (menor=mejor):")
        for r in order:
            print(f"    c{r}: {dists[r]:.4f}")

if __name__ == "__main__":
    main()
