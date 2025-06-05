import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def extrair_features(numero):
    return {
        "numero": numero,
        "cor": 1 if numero % 2 == 0 else 0,  # 1 = vermelho, 0 = preto (simplificado)
        "coluna": numero % 3,
        "linha": numero // 3,
        "range": 1 if numero > 18 else 0,
        "terminal": numero % 10,
        "par": 1 if numero % 2 == 0 else 0,
        "vizinho_anterior": (numero - 1) % 37,
        "vizinho_posterior": (numero + 1) % 37
    }

def prever_proximos_numeros_com_ia(caminho_arquivo, qtd=10):
    try:
        df = pd.read_csv(caminho_arquivo)
        df = df[df["numero"].notnull()]
        df["numero"] = df["numero"].astype(int)

        if len(df) < 50:
            return None

        # Cria dataset com features
        features = [extrair_features(n) for n in df["numero"]]
        dataset = pd.DataFrame(features)
        X = dataset.drop(columns=["numero"])
        y = dataset["numero"]

        # Divide em treino/teste e treina modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = RandomForestClassifier(n_estimators=200, random_state=42)
        modelo.fit(X_train, y_train)

        # Cria todos os 37 números possíveis como candidatos
        candidatos = [extrair_features(n) for n in range(37)]
        df_candidatos = pd.DataFrame(candidatos)
        probabilidades = modelo.predict_proba(df_candidatos)
        top_indices = np.argsort(probabilidades.max(axis=1))[::-1][:qtd]
        melhores = df_candidatos.iloc[top_indices]

        resultados = []
        for i, row in melhores.iterrows():
            numero = row["numero"]
            resultados.append({
                "numero": numero,
                "cor": "vermelho" if row["cor"] == 1 else "preto",
                "coluna": int(row["coluna"]) + 1,
                "linha": int(row["linha"]),
                "range": "alto" if row["range"] == 1 else "baixo",
                "terminal": int(row["terminal"]),
                "vizinho_anterior": int(row["vizinho_anterior"]),
                "vizinho_posterior": int(row["vizinho_posterior"])
            })

        return resultados

    except Exception as e:
        print("Erro ao prever:", e)
        return None
