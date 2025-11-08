#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini demo de integración de comandos por voz con acciones del agente.

Uso:
    python agent_command_demo.py --model knn_voice.joblib --wav comando.wav

Lee un JSON de conteos (sample_counts.json) con el esquema:
{
  "tornillos": 3,
  "clavos": 2,
  "arandelas": 4,
  "tuercas": 1
}
Si no existe, usa un ejemplo por defecto.

Requisitos:
    pip install numpy joblib soundfile librosa
"""
import argparse
import json
import subprocess
from joblib import load
import soundfile as sf
import numpy as np
import librosa

from pathlib import Path

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

def predict_command(wav_path, model_path):
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
    return pred

def run_bayes(counts):
    cmd = ["python", "bayes_estimator.py"] + list(map(str, counts))
    subprocess.run(cmd, check=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="knn_voice.joblib")
    ap.add_argument("--wav", required=True, help="archivo .wav con el comando grabado")
    ap.add_argument("--counts_json", default="sample_counts.json",
                    help="JSON con conteos {tornillos,clavos,arandelas,tuercas}")
    arg
