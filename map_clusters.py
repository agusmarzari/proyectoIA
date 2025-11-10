#!/usr/bin/env python3
import csv, json, os
from collections import Counter, defaultdict

# Asumimos que hiciste una carpeta "labeled/{clase}" con algunas imágenes por clase
LABELED_DIR = "out_clusters_mejorado"
CSV_ASSIGNS = "cluster_assignments.csv"
OUT_JSON = "cluster_to_class.json"

def main():
    # filename -> true class (si existe en labeled/)
    file2class = {}
    for cname in ["tornillos","clavos","arandelas","tuercas"]:
        cdir = os.path.join(LABELED_DIR, cname)
        if not os.path.isdir(cdir): 
            continue
        for f in os.listdir(cdir):
            file2class[f] = cname

    votes = defaultdict(list)
    with open(CSV_ASSIGNS, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            fname = row["filename"]
            cl = int(row["cluster"])
            if fname in file2class:
                votes[cl].append(file2class[fname])

    mapping = {}
    for cl, lablist in votes.items():
        if lablist:
            mapping[cl] = Counter(lablist).most_common(1)[0][0]

    with open(OUT_JSON,"w",encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print("Mapping guardado en", OUT_JSON)

if __name__ == "__main__":
    main()
