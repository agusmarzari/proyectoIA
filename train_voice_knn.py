#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

AUDIO_DIR = "audio"
VOICE_MODEL_PATH = "knn_voice.joblib"

def extract_features(file_path, sr_target=16000, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=sr_target, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        feat = np.mean(mfcc, axis=1)  # promedio temporal
        return feat
    except Exception as e:
        print(f"[Error] {file_path}: {e}")
        return None

X, y = [], []
labels = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
labels.sort()

for label in labels:
    path = os.path.join(AUDIO_DIR, label)
    files = glob.glob(os.path.join(path, "*.wav"))
    for f in files:
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(label)
        else:
            print(f"[Aviso] No se procesó {f}")

X = np.array(X)
y = np.array(y)
print(f"Total audios procesados: {len(X)} | Clases: {sorted(set(y))}")

if len(set(y)) < 2:
    raise SystemExit("Necesito al menos 2 clases en audio/ (contar, salir, proporcion).")

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

dump(model, VOICE_MODEL_PATH)
print(f"[OK] Modelo de voz guardado en {VOICE_MODEL_PATH}")
