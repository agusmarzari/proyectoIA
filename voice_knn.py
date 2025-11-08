#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-NN para reconocimiento de comandos de voz (proporción, contar, salir).

Estructura esperada de datos (entrenamiento):
audio/
  proporcion/
  contar/
  salir/

Comandos:
  Entrenar:
    python voice_knn.py train --data_dir audio --model_out knn_voice.joblib --k 5

  Predecir un WAV:
    python voice_knn.py predict --wav path.wav --model knn_voice.joblib

  Demo con micrófono (opcional, requiere 'sounddevice'):
    python voice_knn.py demo --model knn_voice.joblib --seconds 1.5 --rate 16000

Requisitos:
  pip install librosa soundfile scikit-learn numpy scipy sounddevice
"""
import os
import argparse
import glob
import numpy as np
import soundfile as sf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

import librosa

LABELS = ["proporcion", "contar", "salir"]

def extract_features(y, sr):
    target_len = int(0.8 * sr)
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(mat):
        return np.hstack([mat.mean(axis=1), mat.std(axis=1)])

    feat_mfcc = stats(mfcc)
    feat_d1 = stats(d1)
    feat_d2 = stats(d2)

    S, phase = librosa.magphase(librosa.stft(y, n_fft=512, hop_length=160, win_length=400))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    feat = np.hstack([feat_mfcc, feat_d1, feat_d2, centroid, rolloff, zcr])
    return feat.astype(np.float32)

def load_dataset(data_dir):
    X, y = [], []
    for label in LABELS:
        folder = os.path.join(data_dir, label)
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        if len(files) == 0:
            print(f"ADVERTENCIA: no hay .wav en {folder}")
        for f in files:
            try:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
                feat = extract_features(audio, sr)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print("WARN:", f, e)
    if len(X) == 0:
        raise SystemExit("No se cargaron muestras. Revisa audio/{proporcion,contar,salir}/*.wav")
    return np.array(X), np.array(y)

def train(data_dir, model_out, k=5, test_size=0.2, seed=42):
    X, y = load_dataset(data_dir)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, metric="euclidean"))
    clf.fit(X_tr, y_tr)
    y_pr = clf.predict(X_te)
    print("\nMatriz de confusión:\n", confusion_matrix(y_te, y_pr, labels=LABELS))
    print("\nReporte de clasificación:\n", classification_report(y_te, y_pr, digits=3))
    dump({"pipeline": clf, "labels": LABELS}, model_out)
    print(f"\nModelo guardado en: {model_out}")

def predict(wav_path, model_path):
    bundle = load(model_path)
    clf = bundle["pipeline"]
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    feat = extract_features(audio, sr).reshape(1, -1)
    pred = clf.predict(feat)[0]
    print(pred)
    return pred

def demo(model_path, seconds=1.5, rate=16000):
    import sounddevice as sd
    print(f"Grabando {seconds} s… habla una palabra: 'proporción', 'contar' o 'salir'")
    rec = sd.rec(int(seconds*rate), samplerate=rate, channels=1, dtype='float32')
    sd.wait()
    audio = rec.squeeze()
    bundle = load(model_path)
    clf = bundle["pipeline"]
    feat = extract_features(audio, rate).reshape(1, -1)
    pred = clf.predict(feat)[0]
    print("Predicción:", pred)
    return pred

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train")
    ap_train.add_argument("--data_dir", required=True)
    ap_train.add_argument("--model_out", default="knn_voice.joblib")
    ap_train.add_argument("--k", type=int, default=5)
    ap_train.add_argument("--test_size", type=float, default=0.2)

    ap_pred = sub.add_parser("predict")
    ap_pred.add_argument("--wav", required=True)
    ap_pred.add_argument("--model", required=True)

    ap_demo = sub.add_parser("demo")
    ap_demo.add_argument("--model", required=True)
    ap_demo.add_argument("--seconds", type=float, default=1.5)
    ap_demo.add_argument("--rate", type=int, default=16000)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.data_dir, args.model_out, k=args.k, test_size=args.test_size)
    elif args.cmd == "predict":
        predict(args.wav, args.model)
    elif args.cmd == "demo":
        demo(args.model, seconds=args.seconds, rate=args.rate)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
