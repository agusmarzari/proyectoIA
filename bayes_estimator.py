#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimación bayesiana de la composición de la caja (según consigna del TP).
Calcula P(caja | muestra) para las 4 cajas posibles (a, b, c, d) y devuelve:
- Posterior sobre cada caja
- Proporciones esperadas (promedio bayesiano) por tipo de pieza
- Predicción de cantidades en una caja de 1000 piezas
Uso:
    python bayes_estimator.py 3 2 4 1
    #            tornillos clavos arandelas tuercas  (muestra de 10)
También podés ejecutar sin argumentos y te pedirá los valores por input().
"""
import math
import argparse
from typing import List, Dict

# Orden de clases
CLASSES = ["tornillos", "clavos", "arandelas", "tuercas"]

# Definición de cajas (proporciones verdaderas)
BOXES = {
    "a": [0.25, 0.25, 0.25, 0.25],
    "b": [0.15, 0.30, 0.30, 0.25],
    "c": [0.25, 0.35, 0.25, 0.15],
    "d": [0.50, 0.50, 0.00, 0.00],
}

# Prior uniforme sobre las cajas
PRIOR = {k: 1.0/len(BOXES) for k in BOXES.keys()}

# Pequeño epsilon para robustez numérica (si querés permitir ruido en cajas con 0)
# Si querés que sea "estricto", poné EPS=0.0
EPS = 1e-9

def log_likelihood(counts: List[int], probs: List[float], eps: float = EPS) -> float:
    """Log-likelihood multinomial (omitiendo la constante combinatoria, irrelevante para comparar cajas)."""
    ll = 0.0
    for c, p in zip(counts, probs):
        p_eff = max(p, eps)  # evita log(0)
        if c == 0:
            continue
        ll += c * math.log(p_eff)
    return ll

def posterior_over_boxes(counts: List[int]) -> Dict[str, float]:
    """Devuelve un diccionario con P(box | data) normalizado."""
    logs = {}
    for box_name, probs in BOXES.items():
        ll = log_likelihood(counts, probs, EPS)
        logs[box_name] = math.log(PRIOR[box_name]) + ll

    # normalizar de log-espacio
    m = max(logs.values())
    exps = {k: math.exp(v - m) for k, v in logs.items()}
    Z = sum(exps.values())
    return {k: v / Z for k, v in exps.items()}

def expected_proportions(posterior: Dict[str, float]) -> List[float]:
    """Proporciones esperadas de cada clase, marginalizando sobre las cajas."""
    K = len(CLASSES)
    exp_p = [0.0]*K
    for box_name, w in posterior.items():
        probs = BOXES[box_name]
        for i in range(K):
            exp_p[i] += w * probs[i]
    return exp_p

def pretty_table(headers: List[str], rows: List[List[str]], sep: str = "  ") -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    def fmt_row(row):
        return sep.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
    out = [fmt_row(headers), fmt_row(["-"*w for w in widths])]
    out += [fmt_row(r) for r in rows]
    return "\n".join(out)

def main():
    parser = argparse.ArgumentParser(description="Estimación bayesiana de la composición de la caja")
    parser.add_argument("counts", nargs="*", type=int, help="cuentas de muestra en orden: tornillos clavos arandelas tuercas")
    args = parser.parse_args()

    if args.counts and len(args.counts) != 4:
        raise SystemExit("Debes pasar 4 enteros: tornillos clavos arandelas tuercas")

    if args.counts:
        counts = args.counts
    else:
        counts = []
        for name in CLASSES:
            v = int(input(f"Cantidad en muestra de '{name}': "))
            counts.append(v)

    if sum(counts) == 0:
        raise SystemExit("La muestra no puede ser vacía.")
    print(f"Muestra observada (n={sum(counts)}): {dict(zip(CLASSES, counts))}")

    post = posterior_over_boxes(counts)

    # Mostrar posterior por caja
    rows = []
    for box_name in sorted(post.keys()):
        rows.append([box_name, f"{post[box_name]*100:6.2f}%"])
    print("\nPosterior P(caja | datos):")
    print(pretty_table(["Caja", "Posterior"], rows))

    # Caja MAP
    map_box = max(post.items(), key=lambda kv: kv[1])[0]
    print(f"\nCaja más probable (MAP): {map_box}")

    # Proporciones esperadas y predicción en 1000 piezas
    exp_p = expected_proportions(post)
    pred_1000 = [int(round(1000 * p)) for p in exp_p]

    rows = []
    for name, p, n in zip(CLASSES, exp_p, pred_1000):
        rows.append([name, f"{p*100:6.2f}%", f"{n:4d}"])
    print("\nProporciones esperadas (promedio bayesiano) y predicción en 1000 piezas:")
    print(pretty_table(["Clase", "Proporción", "Predicción/1000"], rows))

    # También mostramos la configuración exacta de la caja MAP
    rows = []
    for name, p in zip(CLASSES, BOXES[map_box]):
        rows.append([name, f"{p*100:6.2f}%", f"{int(round(1000*p)):4d}"])
    print(f"\nSi la caja fuera realmente '{map_box}' (configuración exacta):")
    print(pretty_table(["Clase", "Proporción", "Cantidad/1000"], rows))

if __name__ == "__main__":
    main()
