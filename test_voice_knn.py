from voice_knn import demo
for cmd in ["contar", "proporcion", "salir"]:
    print(f"\n➡️ Di claramente: {cmd.upper()}")
    pred = demo("knn_voice.joblib", seconds=2.0)
    print(f"Predijo: {pred}")
