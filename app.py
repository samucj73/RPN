
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

    timestamps_existentes = {item['timestamp'] for item in dados_existentes}

    novos_filtrados = [item for item in history if item['timestamp'] not in timestamps_existentes]

    if novos_filtrados:
        logging.info(f"Adicionando {len(novos_filtrados)} novos resultados ao arquivo.")
    else:
        logging.info("Nenhum novo resultado para adicionar.")

    dados_existentes.extend(novos_filtrados)
    dados_existentes.sort(key=lambda x: x['timestamp'])

    with open(caminho, "w") as f:
        json.dump(dados_existentes, f, indent=2)

# --- Código da IA ---

def get_color(n):
    if n == 0:
        return -1  # Verde
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
        numero % 2,                      # Par/Ímpar do número
        numero % 3,                      # Resto por 3
        1 if 19 <= numero <= 36 else 0, # Alto/baixo
        get_color(numero),               # Cor
        get_coluna(numero),              # Coluna
        get_linha(numero),               # Linha
        freq_norm.get(numero, 0),       # Frequência normalizada
        (numero - janela[idx_num-1]) if idx_num > 0 else 0,  # Diferença para o anterior
        total_pares / len(janela),      # Proporção pares na janela
        total_impares / len(janela),    # Proporção ímpares na janela
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
        self.classes_ = np.array(list(range(37)))  # 0 a 36
        self.iniciado = False

    def treinar(self, entradas, saidas):
        X = np.array(entradas)
        y = np.array(saidas)
        if not self.iniciado:
            self.modelo.partial_fit(X, y, classes=self.classes_)
            self.iniciado = True
        else:
            self.modelo.partial_fit(X, y)

    def prever(self, entrada, top_k=6, prob_threshold=0.05):
        if not self.iniciado:
            return []
        proba = self.modelo.predict_proba([entrada])[0]
        candidatos = [(idx, p) for idx, p in enumerate(proba) if p >= prob_threshold]
        candidatos.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, p in candidatos[:top_k]]
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

# --- Streamlit interface ---

st.set_page_config(page_title="Roleta IA", layout="wide")
st.title("🎯 Previsão Inteligente de Roleta")

count = st_autorefresh(interval=40000, limit=None, key="auto_refresh")

# Carregar histórico salvo ou iniciar
if "historico" not in st.session_state:
    if os.path.exists(HISTORICO_PATH):
        with open(HISTORICO_PATH, "r") as f:
            try:
                st.session_state.historico = json.load(f)
            except json.JSONDecodeError:
                st.session_state.historico = []
    else:
        st.session_state.historico = []

if "acertos" not in st.session_state:
    st.session_state.acertos = []

if "previsoes" not in st.session_state:
    st.session_state.previsoes = []

# Instância da IA persistida
if "roleta_ia" not in st.session_state:
    st.session_state.roleta_ia = RoletaIA()

resultado = fetch_latest_result()

ultimo_timestamp = (
    st.session_state.historico[-1]["timestamp"] if st.session_state.historico else None
)

if resultado:
    if resultado["timestamp"] != ultimo_timestamp:
        novo_resultado = {
            "number": resultado["number"],
            "color": resultado["color"],
            "timestamp": resultado["timestamp"],
            "lucky_numbers": resultado["lucky_numbers"]
        }
        st.session_state.historico.append(novo_resultado)
        salvar_resultado_em_arquivo([novo_resultado])

        st.toast(f"🆕 Novo número capturado: **{novo_resultado['number']}** ({novo_resultado['color']})", icon="🎲")

        previsoes = st.session_state.roleta_ia.prever_numeros(st.session_state.historico)
        st.session_state.previsoes = previsoes

        if previsoes and resultado["number"] in previsoes:
            if resultado["number"] not in st.session_state.acertos:
                st.session_state.acertos.append(resultado["number"])
                st.toast(f"🎯 Acerto! Número {resultado['number']} estava na previsão!", icon="✅")
    else:
        st.info("⏳ Aguardando novo sorteio...")
else:
    st.error("❌ Falha ao obter dados da API.")

st.subheader("🧾 Últimos Sorteios (números)")
st.write([h["number"] for h in st.session_state.historico[-10:]])

if st.session_state.historico:
    ultimo = st.session_state.historico[-1]
    st.caption(f"⏰ Último sorteio registrado: {ultimo['timestamp']}")

st.subheader("🔮 Previsão de Próximos Números Mais Prováveis")
if st.session_state.previsoes:
    st.success(f"Números Prováveis: {st.session_state.previsoes}")
else:
    st.warning("Aguardando pelo menos 20 sorteios válidos para iniciar previsões.")

st.subheader("🏅 Números Acertados pela IA")
col1, col2 = st.columns([4, 1])

with col1:
    if st.session_state.acertos:
        st.success(f"Números acertados até agora: {st.session_state.acertos}")
    else:
        st.info("Nenhum acerto registrado ainda.")

with col2:
    if st.button("🔄 Resetar Acertos"):
        st.session_state.acertos = []
        st.toast("Acertos resetados com sucesso!", icon="🧹")

st.subheader("📈 Taxa de Acertos da IA")
total_previsoes_possiveis = len([
    h for h in st.session_state.historico if h["number"] not in (None, 0)
]) - 18

total_acertos = len(st.session_state.acertos)

if total_previsoes_possiveis > 0:
    taxa_acerto = (total_acertos / total_previsoes_possiveis) * 100
    st.info(f"🎯 Taxa de acerto da IA: **{taxa_acerto:.2f}%** ({total_acertos} acertos em {total_previsoes_possiveis} previsões)")
else:
    st.warning("🔎 Taxa de acertos será exibida após 20 sorteios.")

with st.expander("📜 Ver histórico completo"):
    st.json(st.session_state.historico)

with st.expander("📂 Ver conteúdo bruto salvo (JSON)"):
    if os.path.exists(HISTORICO_PATH):
        with open(HISTORICO_PATH, "r") as f:
            st.code(f.read(), language="json")
    else:
        st.info("Nenhum histórico salvo ainda.")

st.markdown("---")
st.caption("🔁 Atualiza automaticamente a cada 40 segundos.")
st.caption("🤖 Desenvolvido com aprendizado de máquina online via `SGDClassifier`.")
