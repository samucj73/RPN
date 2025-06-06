import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def extrair_caracteristicas(numero):
    numero = int(numero)
    cor = (
        "vermelho" if numero in [
            1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36
        ] else "preto" if numero != 0 else "verde"
    )
    coluna = (numero - 1) % 3 + 1 if numero != 0 else 0
    linha = ((numero - 1) // 3 + 1) if numero != 0 else 0
    tipo = "baixo" if 1 <= numero <= 18 else "alto" if 19 <= numero <= 36 else "zero"
    terminal = numero % 10
    vizinho_anterior = numero - 1 if numero > 0 else 36
    vizinho_posterior = numero + 1 if numero < 36 else 0

    return {
        "numero": numero,
        "cor": cor,
        "coluna": coluna,
        "linha": linha,
        "range": tipo,
        "terminal": terminal,
        "vizinho_anterior": vizinho_anterior,
        "vizinho_posterior": vizinho_posterior
    }

def preparar_dados(df):
    df = df.copy()
    df = df[df['numero'] != '']
    df['numero'] = df['numero'].astype(int)

    features = df['numero'].apply(extrair_caracteristicas).apply(pd.Series)
    df = pd.concat([df, features], axis=1)

    encoders = {}
    for col in ['cor', 'range']:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        encoders[col] = enc

    X = df[['cor', 'coluna', 'linha', 'range', 'terminal', 'vizinho_anterior', 'vizinho_posterior']]
    y = df['numero']
    return X, y, encoders

def prever_proximos_numeros_com_ia(caminho_csv, qtd=5):
    try:
        df = pd.read_csv(caminho_csv)
        if len(df) < 10:  # ✅ Limite mínimo reduzido para facilitar teste
            return []

        X, y, encoders = preparar_dados(df)
        modelo = RandomForestClassifier(n_estimators=200, random_state=42)
        modelo.fit(X, y)

        previsoes = []
        for _ in range(qtd):
            ultimo_numero = df['numero'].astype(int).iloc[-1]
            ult_feat = extrair_caracteristicas(ultimo_numero)

            entrada = pd.DataFrame([{
                'cor': encoders['cor'].transform([ult_feat['cor']])[0],
                'coluna': ult_feat['coluna'],
                'linha': ult_feat['linha'],
                'range': encoders['range'].transform([ult_feat['range']])[0],
                'terminal': ult_feat['terminal'],
                'vizinho_anterior': ult_feat['vizinho_anterior'],
                'vizinho_posterior': ult_feat['vizinho_posterior']
            }])

            pred = modelo.predict(entrada)[0]
            previsao_completa = extrair_caracteristicas(pred)
            previsoes.append(previsao_completa)

        return previsoes

    except Exception as e:
        print(f"[ERRO IA] Falha na previsão: {e}")
        return []
