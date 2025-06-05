import pandas as pd
import random

def prever_proximos_numeros_com_ia(caminho_arquivo, qtd=10):
    try:
        df = pd.read_csv(caminho_arquivo)
        numeros = df["numero"].dropna().astype(int).tolist()
        if len(numeros) < 30:
            return None
        resultados = []
        for _ in range(qtd):
            numero = random.randint(0, 36)
            resultados.append({
                "numero": numero,
                "cor": "vermelho" if numero % 2 == 0 else "preto",
                "coluna": (numero % 3) + 1,
                "linha": (numero // 3) + 1,
                "range": "alto" if numero > 18 else "baixo",
                "terminal": numero % 10,
                "vizinho_anterior": (numero - 1) % 37,
                "vizinho_posterior": (numero + 1) % 37
            })
        return resultados
    except Exception as e:
        print("Erro ao prever:", e)
        return None
