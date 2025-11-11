#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera y guarda la máscara binaria de una imagen del dataset.
Mostrará el objeto original y la máscara resultante.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Ruta de una imagen del dataset (cambiá el nombre por una tuya)
img_path = "dataset_mix\WhatsApp Image 2025-11-10 at 09.09.55.jpeg"

# Leer imagen y convertir a escala de grises
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Normalizar contraste y suavizar ruido
gray_eq = cv2.equalizeHist(gray)
blur = cv2.GaussianBlur(gray_eq, (5,5), 0)

# Umbral adaptativo (objeto oscuro sobre fondo claro)
_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Operaciones morfológicas (limpieza)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

# Buscar contornos y quedarse con el más grande
cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(th)
if cnts:
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)

# Guardar la máscara en disco
cv2.imwrite("mask_demo.png", mask)
print("✅ Máscara guardada como mask_demo.png")

# Mostrar resultado
plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Imagen original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(mask, cmap="gray"); plt.title("Máscara binaria"); plt.axis("off")
plt.show()
