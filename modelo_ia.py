
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def prever_proximos_numeros_com_ia(caminho_arquivo, qtd=10):
    try:
        df = pd.read_csv(caminho_arquivo)
        df = df.dropna()
        if len(df) < 10:
            return None

        df["numero"] = df["numero"].astype(int)
        df["anterior"] = df["numero"].shift(1).fillna(method='bfill')
        df["posterior"] = df["numero"].shift(-1).fillna(method='ffill')
        df = df.dropna()

        X = df[["anterior", "posterior"]]
        y = df["numero"]

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)

        ult = df.iloc[-1]
        entrada = [[ult["anterior"], ult["posterior"]]]

        resultados = []
        for _ in range(qtd):
            numero = modelo.predict(entrada)[0]
            resultados.append({
                "numero": int(numero),
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
        print("Erro na previsão IA:", e)
        return None
