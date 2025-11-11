#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrena un modelo de reconocimiento de tres comandos de voz (contar, salir, proporcion),
sin grabar datos nuevos:
- Features: MFCC (13) + delta + delta-delta -> 39 por frame
- Agregación: media y desvío por coeficiente -> 78 dims
- StandardScaler
- Model selection: KNN (weights='distance') vs SVM RBF (pequeña grid)
- (Opcional) Aumentación sintética: pitch ±1, speed 0.95/1.05, ruido leve
Guarda todo en voice_model.joblib
"""

import os, glob, random
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from joblib import dump

# ---------------- Config ----------------
AUDIO_DIR   = "audio"
SR          = 16000
N_MFCC      = 13
USE_AUG     = True          # ponelo en False si no querés aumentación sintética
AUG_MULT    = 3             # cuántas variantes (aprox) generar por audio original
RNG_SEED    = 42
# ---------------------------------------

rng = np.random.default_rng(RNG_SEED)

def load_wav(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # recorte suave de silencios extremos (no obligatorio, ayuda si hay silencios largos)
    y, _ = librosa.effects.trim(y, top_db=25)
    return y

def extract_feat(y, sr=SR, n_mfcc=N_MFCC):
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1    = librosa.feature.delta(mfcc)
    d2    = librosa.feature.delta(mfcc, order=2)
    M     = np.vstack([mfcc, d1, d2])   # (39, T)
    mu    = M.mean(axis=1)
    sd    = M.std(axis=1) + 1e-9
    feat  = np.hstack([mu, sd]).astype(np.float32)  # (78,)
    return feat

# ----- Aumentaciones ligeras -----
def aug_pitch(y, sr=SR, semitones=1.0):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

def aug_speed(y, factor=1.05):
    # cambiar velocidad re-muestreando con resample
    idx = np.arange(0, len(y), factor)
    idx = idx[idx < len(y)].astype(int)
    return y[idx]

def aug_noise(y, snr_db=25.0):
    # añade ruido blanco para SNR deseada (aprox)
    sig_pow = np.mean(y**2) + 1e-12
    snr = 10**(snr_db/10.0)
    noise_pow = sig_pow / snr
    n = rng.normal(0.0, np.sqrt(noise_pow), size=y.shape)
    return y + n.astype(y.dtype)

def gen_augments(y):
    outs = []
    # combinaciones pequeñas
    choices = [
        ("pitch", +1.0),
        ("pitch", -1.0),
        ("speed", 1.05),
        ("speed", 0.95),
        ("noise", 25.0),
    ]
    rng.shuffle(choices)
    for i in range(min(AUG_MULT, len(choices))):
        kind, val = choices[i]
        if kind == "pitch":
            outs.append(aug_pitch(y, SR, semitones=val))
        elif kind == "speed":
            outs.append(aug_speed(y, factor=val))
        elif kind == "noise":
            outs.append(aug_noise(y, snr_db=val))
    return outs

# ----- Cargar dataset -----
def load_dataset():
    X, y = [], []
    labels = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
    labels = sorted(labels)
    if not labels:
        raise SystemExit(f"No encontré subcarpetas en {AUDIO_DIR}. Deben ser: contar, salir, proporcion")

    # 1) Cargo originales (y, si USE_AUG=True, ya agrego algo de aug global)
    per_class_raw = {lab: [] for lab in labels}
    for lab in labels:
        files = glob.glob(os.path.join(AUDIO_DIR, lab, "*.wav"))
        for f in files:
            try:
                wav = load_wav(f)
                per_class_raw[lab].append(wav)
                X.append(extract_feat(wav)); y.append(lab)
                if USE_AUG:
                    for aug in gen_augments(wav):
                        X.append(extract_feat(aug)); y.append(lab)
            except Exception as e:
                print(f"[ERROR] {f}: {e}")

    # 2) Oversampling dirigido para 'proporcion' (sin audios nuevos)
    target_cls = "proporcion"
    if target_cls in per_class_raw:
        # cuántos ejemplos tiene cada clase en X,y (tras la primera pasada)
        from collections import Counter
        cnt = Counter(y)
        max_count = max(cnt.values())
        cur = cnt.get(target_cls, 0)
        # generamos variantes hasta igualar a la clase mayor
        while cur < max_count:
            wav = random.choice(per_class_raw[target_cls])
            for aug in gen_augments(wav):
                X.append(extract_feat(aug)); y.append(target_cls)
                cur += 1
                if cur >= max_count:
                    break

    X = np.array(X); y = np.array(y)
    if len(set(y)) < 2:
        raise SystemExit("Necesito al menos 2 clases en audio/")

    print(f"Total ejemplos: {len(X)} | Clases: {sorted(set(y))} | Dim: {X.shape[1]}")
    from collections import Counter
    print("Distribución post-balanceo:", Counter(y))
    return X, y, labels

def main():
    X, y, labels = load_dataset()

    # Modelos candidatos
    knn_params = [3, 5, 7]
    models = []
    for k in knn_params:
        models.append(("KNN(k=%d, dist)" % k, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights='distance'))))
    # SVM RBF (grid chicas) con balance de clases
    for C in [1, 5, 10]:
        for gamma in ["scale", 0.01, 0.001]:
            models.append((
                f"SVM(C={C},gamma={gamma})",
                make_pipeline(StandardScaler(), SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced'))
        ))


    # Selección por CV estratificada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    best_name, best_model, best_score = None, None, -1.0

    print("\n== Evaluación CV (5-fold) ==")
    for name, mdl in models:
        scores = cross_val_score(mdl, X, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        print(f"{name:24s}  acc={mean_acc:.3f}  (+/- {scores.std():.3f})")
        if mean_acc > best_score:
            best_score = mean_acc
            best_model = mdl
            best_name  = name

    print(f"\n→ Mejor: {best_name}  acc_cv={best_score:.3f}")

    # Reentrenar el mejor con TODO el dataset
    best_model.fit(X, y)

    # Guardar bundle
    artifact = {
        "model": best_model,
        "labels": sorted(set(y)),
        "sr": SR,
        "n_mfcc": N_MFCC,
        "use_deltas": True,
        "agg": "mean_std",
        "version": 1,
    }
    dump(artifact, "voice_model.joblib")
    print("[OK] Guardado voice_model.joblib")

if __name__ == "__main__":
    main()
