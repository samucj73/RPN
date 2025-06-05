import csv
import os
from datetime import datetime
import random

def fetch_latest_result():
    # Simulador de sorteio (substitua por sua lógica de captura real)
    now = datetime.now().isoformat()
    return {
        "timestamp": now,
        "number": random.randint(0, 36),
        "lucky_numbers": [random.randint(0, 36) for _ in range(3)],
    }

def salvar_resultado_em_arquivo(result):
    file_path = "resultados.csv"
    existe = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "numero", "lucky"])
        if not existe:
            writer.writeheader()
        writer.writerow({
            "timestamp": result["timestamp"],
            "numero": result["number"],
            "lucky": "-".join(map(str, result["lucky_numbers"]))
        })
