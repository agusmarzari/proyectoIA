from joblib import load
from pathlib import Path

MODEL = "kmeans_piezas.joblib"   # cambia si hace falta

b = load(MODEL)
print("== PATH ==", Path(MODEL).resolve())
print("== KEYS ==", list(b.keys()))
kmeans = b["kmeans"]
scaler = b["scaler"]
pca = b.get("pca", None)
classes = b.get("classes", [])
c2c = b.get("cluster_to_class", {})

print(f"n_clusters: {kmeans.n_clusters}")
print("classes:", classes)
print("cluster_to_class:", c2c)
print("scaler.n_features_in_:", getattr(scaler, "n_features_in_", "¿?"))
if pca is not None:
    print("pca.n_components_:", getattr(pca, "n_components_", "¿?"))
