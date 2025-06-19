import streamlit as st
import json
import os
import logging
import requests
from collections import Counter
from sklearn.linear_model import SGDClassifier
import numpy as np
from streamlit_autorefresh import st_autorefresh
import re

# --- Configurações ---
HISTORICO_PATH = "historico_resultados.json"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# --- API ---
def fetch_latest_result():
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        game_data = data.get("data", {})
        result = game_data.get("result", {})
        outcome = result.get("outcome", {})
        lucky_list = result.get("luckyNumbersList", [])

        number = outcome.get("number")
        color = outcome.get("color", "-")
        timestamp = game_data.get("startedAt")
        lucky_numbers = [item["number"] for item in lucky_list]

        return {
            "number": number,
            "color": color,
            "timestamp": timestamp,
            "lucky_numbers": lucky_numbers
        }
    except Exception as e:
        logging.error(f"Erro ao buscar resultado da API: {e}")
        return None

def salvar_resultado_em_arquivo(history, caminho=HISTORICO_PATH):
    dados_existentes = []
    if os.path.exists(caminho):
        with open(caminho, "r") as f:
            try:
                dados_existentes = json.load(f)
            except json.JSONDecodeError:
                dados_existentes = []
    timestamps_existentes = {item['timestamp'] for item in dados_existentes if 'timestamp' in item}
    novos_filtrados = [item for item in history if item.get('timestamp') not in timestamps_existentes]
    dados_existentes.extend(novos_filtrados)
    dados_existentes.sort(key=lambda x: x.get('timestamp', 'manual'))
    with open(caminho, "w") as f:
        json.dump(dados_existentes, f, indent=2)

# --- Funções Auxiliares ---
def get_color(n):
    if n == 0: return -1
    return 1 if n in {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36} else 0

def get_coluna(n):
    return (n - 1) % 3 + 1 if n != 0 else 0

def get_linha(n):
    return ((n - 1) // 3) + 1 if n != 0 else 0

def extrair_features(numero, freq_norm, janela, idx_num, total_pares, total_impares):
    return [
        numero % 2,
        numero % 3,
        1 if 19 <= numero <= 36 else 0,
        get_color(numero),
        get_coluna(numero),
        get_linha(numero),
        freq_norm.get(numero, 0),
        (numero - janela[idx_num-1]) if idx_num > 0 else 0,
        total_pares / len(janela),
        total_impares / len(janela),
    ]

def construir_entrada(janela, freq, freq_total):
    freq_norm = {k: v / freq_total for k, v in freq.items()} if freq_total > 0 else {}
    total_pares = sum(1 for n in janela if n != 0 and n % 2 == 0)
    total_impares = sum(1 for n in janela if n != 0 and n % 2 == 1)
    features = []
    for i, n in enumerate(janela):
        features.extend(extrair_features(n, freq_norm, janela, i, total_pares, total_impares))
    return features

# --- Modelo de IA ---
class ModeloIA:
    def __init__(self):
        self.modelo = SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            eta0=0.01,
            learning_rate='optimal',
            random_state=42
        )
        self.classes_ = np.arange(37)
        self.iniciado = False

    def treinar(self, entradas, saidas):
        X = np.array(entradas)
        y = np.array(saidas)
        if not self.iniciado:
            self.modelo.partial_fit(X, y, classes=self.classes_)
            self.iniciado = True
        else:
            self.modelo.partial_fit(X, y)

    def prever(self, entrada, top_k=8):
        if not self.iniciado:
            return []
        proba = self.modelo.predict_proba([entrada])[0]
        indices = np.argsort(proba)[::-1]
        return indices[:top_k].tolist()

class RoletaIA:
    def __init__(self, janela_min=18, janela_max=36):
        self.modelo = ModeloIA()
        self.janela_min = janela_min
        self.janela_max = janela_max

    def treinar_batch(self, numeros):
        entradas, saidas = [], []
        for i in range(self.janela_max, len(numeros) - 1):
            janela = numeros[i - self.janela_max:i]
            saida = numeros[i]
            if any(n < 0 or n > 36 for n in janela + [saida]): continue
            freq = Counter(numeros[:i])
            entrada = construir_entrada(janela, freq, sum(freq.values()))
            entradas.append(entrada)
            saidas.append(saida)
        if entradas and saidas:
            self.modelo.treinar(entradas, saidas)

    def prever_numeros(self, historico):
        numeros = [item["number"] for item in historico]
        if len(numeros) < self.janela_min + 1:
            return {"numeros": [], "coluna": 0, "linha": 0}
        self.treinar_batch(numeros)
        janela = numeros[-self.janela_max:]
        entrada = construir_entrada(janela, Counter(numeros), sum(Counter(numeros).values()))
        previsoes = self.modelo.prever(entrada)
        colunas = [get_coluna(n) for n in previsoes if n != 0]
        linhas = [get_linha(n) for n in previsoes if n != 0]
        return {
            "numeros": previsoes,
            "coluna": max(set(colunas), key=colunas.count) if colunas else 0,
            "linha": max(set(linhas), key=linhas.count) if linhas else 0
        }

# --- Streamlit Interface ---
st.set_page_config(page_title="Roleta IA", layout="wide")
st.title("🎯 Previsão Inteligente de Roleta")

# Inicialização de estado
if "historico" not in st.session_state:
    st.session_state.historico = json.load(open(HISTORICO_PATH)) if os.path.exists(HISTORICO_PATH) else []
if "acertos" not in st.session_state: st.session_state.acertos = []
if "previsoes" not in st.session_state: st.session_state.previsoes = []
if "roleta_ia" not in st.session_state: st.session_state.roleta_ia = RoletaIA()

min_sorteios = st.slider("Quantidade mínima de sorteios para previsão", 5, 100, 18)

# Entrada manual
st.subheader("✍️ Inserir até 100 Sorteios Anteriores Manualmente")
entrada = st.text_area("Cole os números (com ou sem texto, separado por espaço):", height=100)
if st.button("Adicionar Sorteios"):
    numeros_extraidos = [int(n) for n in re.findall(r'\b\d{1,2}\b', entrada) if 0 <= int(n) <= 36]
    if len(numeros_extraidos) > 100:
        st.warning("Insira no máximo 100 números.")
    else:
        for numero in numeros_extraidos:
            st.session_state.historico.append({
                "number": numero,
                "color": "-",
                "timestamp": f"manual_{len(st.session_state.historico)}",
                "lucky_numbers": []
            })
        salvar_resultado_em_arquivo(st.session_state.historico)
        st.success(f"{len(numeros_extraidos)} números adicionados com sucesso!")

# Captura automática
st_autorefresh(interval=40000, key="auto_refresh")
resultado = fetch_latest_result()
ultimo_ts = st.session_state.historico[-1]["timestamp"] if st.session_state.historico else None
if resultado and resultado["timestamp"] != ultimo_ts:
    novo = {
        "number": resultado["number"],
        "color": resultado["color"],
        "timestamp": resultado["timestamp"],
        "lucky_numbers": resultado["lucky_numbers"]
    }
    st.session_state.historico.append(novo)
    salvar_resultado_em_arquivo([novo])
    st.toast(f"🎲 Novo número: {novo['number']}")
    previsao = st.session_state.roleta_ia.prever_numeros(st.session_state.historico)
    st.session_state.previsoes = previsao["numeros"]
    st.session_state.coluna = previsao["coluna"]
    st.session_state.linha = previsao["linha"]
    if novo["number"] in st.session_state.previsoes and novo["number"] not in st.session_state.acertos:
        st.session_state.acertos.append(novo["number"])
        st.toast(f"✅ Acerto! {novo['number']} estava na previsão.")
else:
    st.info("⏳ Aguardando novo sorteio...")

# Exibição
st.subheader("🔢 Últimos Sorteios")
st.write(" ".join(str(h["number"]) for h in st.session_state.historico[-10:]))

st.subheader("🔮 Previsão dos Próximos 8 Números")
if st.session_state.previsoes:
    st.success("🎯 " + " ".join(str(n) for n in st.session_state.previsoes))
    st.info(f"📐 Coluna provável: {st.session_state.coluna} | Linha provável: {st.session_state.linha}")
else:
    st.warning("Aguardando sorteios suficientes.")

st.subheader("🏅 Acertos")
col1, col2 = st.columns([4, 1])
with col1:
    st.success(" ".join(str(n) for n in st.session_state.acertos) if st.session_state.acertos else "Nenhum acerto.")
with col2:
    if st.button("Resetar Acertos"):
        st.session_state.acertos = []
        st.toast("Acertos resetados.")

st.subheader("📊 Taxa de Acertos")
total_prev = len([h for h in st.session_state.historico if h["number"] not in (None, 0)]) - min_sorteios
if total_prev > 0:
    taxa = len(st.session_state.acertos) / total_prev * 100
    st.info(f"🎯 Taxa de acerto: {taxa:.2f}%")
else:
    st.warning("Aguardando mais sorteios para calcular.")
