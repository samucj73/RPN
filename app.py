
import streamlit as st
import json
import os
import logging
import requests
from collections import Counter
from sklearn.linear_model import SGDClassifier
import numpy as np
from streamlit_autorefresh import st_autorefresh

# --- Configurações ---

HISTORICO_PATH = "historico_resultados.json"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# --- Funções da API ---

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
                logging.warning("Arquivo JSON vazio ou corrompido. Recriando arquivo.")
                dados_existentes = []

    timestamps_existentes = {item['timestamp'] for item in dados_existentes if 'timestamp' in item}

    novos_filtrados = [item for item in history if item.get('timestamp') not in timestamps_existentes]

    dados_existentes.extend(novos_filtrados)
    dados_existentes.sort(key=lambda x: x.get('timestamp', 'manual'))

    with open(caminho, "w") as f:
        json.dump(dados_existentes, f, indent=2)

# --- Código da IA ---
def get_color(n):
    if n == 0:
        return -1
    return 1 if n in {
        1, 3, 5, 7, 9, 12, 14, 16, 18,
        19, 21, 23, 25, 27, 30, 32, 34, 36
    } else 0

def get_coluna(n):
    return (n - 1) % 3 + 1 if n != 0 else 0

def get_linha(n):
    return ((n - 1) // 3) + 1 if n != 0 else 0

def extrair_features(numero, freq_norm, janela, idx_num, total_pares, total_impares):
    features = [
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
    return features

def construir_entrada(janela, freq, freq_total):
    freq_norm = {k: v / freq_total for k, v in freq.items()} if freq_total > 0 else {}
    total_pares = sum(1 for n in janela if n != 0 and n % 2 == 0)
    total_impares = sum(1 for n in janela if n != 0 and n % 2 == 1)
    features = []
    for i, n in enumerate(janela):
        feats = extrair_features(n, freq_norm, janela, i, total_pares, total_impares)
        features.extend(feats)
    return features

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
        self.classes_ = np.array(list(range(37)))
        self.iniciado = False

    def treinar(self, entradas, saidas):
        X = np.array(entradas)
        y = np.array(saidas)
        if not self.iniciado:
            self.modelo.partial_fit(X, y, classes=self.classes_)
            self.iniciado = True
        else:
            self.modelo.partial_fit(X, y)

    def prever(self, entrada, top_k=4, prob_threshold=0.01):
        if not self.iniciado:
            return []
        proba = self.modelo.predict_proba([entrada])[0]
        candidatos = [(idx, p) for idx, p in enumerate(proba) if p >= prob_threshold]
        candidatos.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, p in candidatos[:top_k]]
        if len(top_indices) < top_k:
            top_restantes = np.argsort(proba)[::-1]
            for idx in top_restantes:
                if idx not in top_indices:
                    top_indices.append(idx)
                if len(top_indices) == top_k:
                    break
        return top_indices

class RoletaIA:
    def __init__(self, janela_min=18, janela_max=36):
        self.modelo = ModeloIA()
        self.janela_min = janela_min
        self.janela_max = janela_max

    def treinar_batch(self, numeros):
        entradas = []
        saidas = []
        for i in range(self.janela_max, len(numeros) - 1):
            janela_tamanho = min(self.janela_max, i)
            janela = numeros[i - janela_tamanho:i]
            saida = numeros[i]
            freq = Counter(numeros[:i])
            freq_total = sum(freq.values())
            entrada = construir_entrada(janela, freq, freq_total)
            entradas.append(entrada)
            saidas.append(saida)
        if entradas and saidas:
            self.modelo.treinar(entradas, saidas)

    def prever_numeros(self, historico):
        numeros = [item["number"] for item in historico]
        if len(numeros) < self.janela_min + 1:
            return []
        self.treinar_batch(numeros)
        janela_recente = numeros[-self.janela_max:]
        freq_final = Counter(numeros[:-1])
        freq_total_final = sum(freq_final.values())
        entrada = construir_entrada(janela_recente, freq_final, freq_total_final)
        return self.modelo.prever(entrada)

# --- Streamlit ---

st.set_page_config(page_title="Roleta IA", layout="wide")
st.title("🎯 Previsão Inteligente de Roleta")

min_sorteios_para_prever = st.slider("Quantidade mínima de sorteios para previsão", 5, 100, 18)

# Entrada manual
st.subheader("✍️ Inserir Sorteios Anteriores Manualmente")
input_numbers = st.text_input("Digite até 6 números separados por vírgula (ex: 23,7,11):")
if st.button("Adicionar Sorteios Manuais"):
    try:
        nums = [int(n.strip()) for n in input_numbers.split(",") if n.strip().isdigit()]
        for numero in nums:
            st.session_state.historico.append({
                "number": numero,
                "color": "-",  # cor desconhecida
                "timestamp": f"manual_{len(st.session_state.historico)}",
                "lucky_numbers": []
            })
        salvar_resultado_em_arquivo(st.session_state.historico)
        st.success("Números adicionados ao histórico com sucesso.")
    except:
        st.error("Erro ao interpretar os números. Use apenas números separados por vírgula.")

count = st_autorefresh(interval=40000, limit=None, key="auto_refresh")

# Session state
if "historico" not in st.session_state:
    if os.path.exists(HISTORICO_PATH):
        with open(HISTORICO_PATH, "r") as f:
            try:
                st.session_state.historico = json.load(f)
            except:
                st.session_state.historico = []
    else:
        st.session_state.historico = []

if "acertos" not in st.session_state:
    st.session_state.acertos = []

if "previsoes" not in st.session_state:
    st.session_state.previsoes = []

if "roleta_ia" not in st.session_state:
    st.session_state.roleta_ia = RoletaIA(janela_min=min_sorteios_para_prever)

# API
resultado = fetch_latest_result()
ultimo_timestamp = (
    st.session_state.historico[-1]["timestamp"] if st.session_state.historico else None
)

if resultado and resultado["timestamp"] != ultimo_timestamp:
    novo_resultado = {
        "number": resultado["number"],
        "color": resultado["color"],
        "timestamp": resultado["timestamp"],
        "lucky_numbers": resultado["lucky_numbers"]
    }
    st.session_state.historico.append(novo_resultado)
    salvar_resultado_em_arquivo([novo_resultado])
    st.toast(f"🎲 Novo número capturado: {novo_resultado['number']}")
    previsoes = st.session_state.roleta_ia.prever_numeros(st.session_state.historico)
    st.session_state.previsoes = previsoes
    if resultado["number"] in previsoes:
        if resultado["number"] not in st.session_state.acertos:
            st.session_state.acertos.append(resultado["number"])
            st.toast(f"✅ Acerto! {resultado['number']} estava na previsão!")
else:
    st.info("⏳ Aguardando novo sorteio...")

# Interface
st.subheader("🧾 Últimos Sorteios")
st.write([h["number"] for h in st.session_state.historico[-10:]])

st.subheader("🔮 Previsão dos Próximos 4 Números")
if st.session_state.previsoes:
    st.success(f"Previsões: {st.session_state.previsoes}")
else:
    st.warning("Aguardando sorteios suficientes para iniciar...")

st.subheader("🏅 Acertos da IA")
col1, col2 = st.columns([4, 1])
with col1:
    st.success(f"Acertos: {st.session_state.acertos}") if st.session_state.acertos else st.info("Nenhum acerto.")
with col2:
    if st.button("Resetar Acertos"):
        st.session_state.acertos = []
        st.toast("Acertos resetados.")

st.subheader("📊 Taxa de Acertos")
total_prev = len([
    h for h in st.session_state.historico if h["number"] not in (None, 0)
]) - min_sorteios_para_prever
if total_prev > 0:
    acertos = len(st.session_state.acertos)
    taxa = acertos / total_prev * 100
    st.info(f"Taxa de acerto: {taxa:.2f}% ({acertos}/{total_prev})")
else:
    st.warning("Taxa será exibida após sorteios suficientes.")
