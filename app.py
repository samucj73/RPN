
import streamlit as st
import json
import os
import logging
import requests
from collections import Counter
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

HISTORICO_PATH = "historico_resultados.json"
MODELO_PATH = "modelo_roleta.joblib"
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

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
        return {"number": number, "color": color, "timestamp": timestamp, "lucky_numbers": lucky_numbers}
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

def get_color(n):
    if n == 0: return -1
    return 1 if n in {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36} else 0

def get_coluna(n): return (n - 1) % 3 + 1 if n != 0 else 0
def get_linha(n): return ((n - 1) // 3) + 1 if n != 0 else 0

def extrair_features(numero, freq_norm, janela, idx_num, total_pares, total_impares, media_geral):
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
        numero - media_geral
    ]

def construir_entrada(janela, freq, freq_total):
    freq_norm = {k: v / freq_total for k, v in freq.items()} if freq_total > 0 else {}
    total_pares = sum(1 for n in janela if n != 0 and n % 2 == 0)
    total_impares = sum(1 for n in janela if n != 0 and n % 2 == 1)
    media_geral = np.mean(janela)
    features = []
    for i, n in enumerate(janela):
        features.extend(extrair_features(n, freq_norm, janela, i, total_pares, total_impares, media_geral))
    return features

class ModeloIA:
    def __init__(self):
        self.modelo = None
        self.treinado = False
        self._carregar_modelo()

    def _carregar_modelo(self):
        if os.path.exists(MODELO_PATH):
            self.modelo = joblib.load(MODELO_PATH)
            self.treinado = True
        else:
            self.modelo = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)

    def treinar(self, entradas, saidas):
        X, y = np.array(entradas), np.array(saidas)
        self.modelo.fit(X, y)
        self.treinado = True
        joblib.dump(self.modelo, MODELO_PATH)

    def prever(self, entrada, top_k=8):
        if not self.treinado: return []
        proba = self.modelo.predict_proba([entrada])[0]
        top_indices = np.argsort(proba)[::-1][:top_k]
        return top_indices.tolist()

class RoletaIA:
    def __init__(self, janela_min=5, janela_max=90):
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
        if entradas: self.modelo.treinar(entradas, saidas)

    def prever_numeros(self, historico):
        numeros = [item["number"] for item in historico]
        if len(numeros) < self.janela_min + 1: return {}
        self.treinar_batch(numeros)
        janela_recente = numeros[-self.janela_max:]
        freq = Counter(numeros[:-1])
        entrada = construir_entrada(janela_recente, freq, sum(freq.values()))
        previsoes = self.modelo.prever(entrada)
        colunas = [get_coluna(n) for n in previsoes if n != 0]
        linhas = [get_linha(n) for n in previsoes if n != 0]
        return {
            "numeros": previsoes,
            "coluna": max(set(colunas), key=colunas.count) if colunas else 0,
            "linha": max(set(linhas), key=linhas.count) if linhas else 0
        }

# Continuação no próximo bloco

# --- Streamlit App ---

st.set_page_config(page_title="Roleta IA", layout="wide")
st.title("🎯 Previsão Inteligente de Roleta")

min_sorteios_para_prever = st.slider("Quantidade mínima de sorteios para previsão", 5, 100, 18)

# Sessões
if "historico" not in st.session_state:
    st.session_state.historico = json.load(open(HISTORICO_PATH)) if os.path.exists(HISTORICO_PATH) else []
if "acertos" not in st.session_state: st.session_state.acertos = []
if "colunas_acertadas" not in st.session_state: st.session_state.colunas_acertadas = 0
if "linhas_acertadas" not in st.session_state: st.session_state.linhas_acertadas = 0
if "previsoes" not in st.session_state: st.session_state.previsoes = []
if "roleta_ia" not in st.session_state: st.session_state.roleta_ia = RoletaIA(janela_min=min_sorteios_para_prever)

# Entrada manual
st.subheader("✍️ Inserir até 100 Sorteios Anteriores Manualmente")
input_numbers = st.text_area("Digite os números separados por espaço:", height=100)
if st.button("Adicionar Sorteios Manuais"):
    try:
        nums = [int(n) for n in input_numbers.split() if n.isdigit() and 0 <= int(n) <= 36]
        if len(nums) > 100:
            st.warning("Você só pode inserir até 100 números.")
        else:
            for numero in nums:
                st.session_state.historico.append({
                    "number": numero, "color": "-", 
                    "timestamp": f"manual_{len(st.session_state.historico)}", 
                    "lucky_numbers": []})
            salvar_resultado_em_arquivo(st.session_state.historico)
            st.success(f"{len(nums)} números adicionados.")
    except:
        st.error("Erro ao interpretar os números.")

st_autorefresh(interval=30000, key="auto_refresh")

# Captura e verificação
resultado = fetch_latest_result()
ultimo_timestamp = st.session_state.historico[-1]["timestamp"] if st.session_state.historico else None

if resultado and resultado["timestamp"] != ultimo_timestamp:
    novo_resultado = resultado
    st.session_state.historico.append(novo_resultado)
    salvar_resultado_em_arquivo([novo_resultado])
    st.toast(f"🎲 Novo número: {novo_resultado['number']}")

    if novo_resultado["number"] in st.session_state.previsoes:
        if novo_resultado["number"] not in st.session_state.acertos:
            st.session_state.acertos.append(novo_resultado["number"])
            st.toast("✅ Acertou o número!")
    if get_coluna(novo_resultado["number"]) == st.session_state.get("coluna_prevista", -1):
        st.session_state.colunas_acertadas += 1
        st.toast("✅ Acertou a coluna!")
    if get_linha(novo_resultado["number"]) == st.session_state.get("linha_prevista", -1):
        st.session_state.linhas_acertadas += 1
        st.toast("✅ Acertou a linha!")

    previsoes = st.session_state.roleta_ia.prever_numeros(st.session_state.historico)
    st.session_state.previsoes = previsoes.get("numeros", [])
    st.session_state.coluna_prevista = previsoes.get("coluna", 0)
    st.session_state.linha_prevista = previsoes.get("linha", 0)

# Interface
st.subheader("🔁 Últimos Sorteios")
st.write(" ".join(str(h["number"]) for h in st.session_state.historico[-10:]))

st.subheader("🔮 Previsão dos Próximos 8 Números")
if st.session_state.previsoes:
    previsao_str = " ".join(f"🎯 {n}" for n in st.session_state.previsoes)
    st.success(f"Números previstos: {previsao_str}")
    st.info(f"🧱 Coluna: {st.session_state.coluna_prevista} | 📐 Linha: {st.session_state.linha_prevista}")
else:
    st.warning("Previsão ainda não disponível.")

st.subheader("🏅 Acertos da IA")
col1, col2 = st.columns([4, 1])
with col1:
    if st.session_state.acertos:
        acertos_str = " ".join(f"✅ {n}" for n in st.session_state.acertos)
        st.success(f"Acertos: {acertos_str}")
    else:
        st.info("Nenhum acerto ainda.")
with col2:
    if st.button("Resetar Acertos"):
        st.session_state.acertos = []
        st.session_state.colunas_acertadas = 0
        st.session_state.linhas_acertadas = 0
        st.toast("Acertos resetados.")

if st.button("🔄 Reiniciar Tudo (Modelo + Histórico)"):
    st.session_state.clear()
    if os.path.exists(HISTORICO_PATH):
        os.remove(HISTORICO_PATH)
    if os.path.exists(MODELO_PATH):
        os.remove(MODELO_PATH)
    st.success("Reiniciado com sucesso. Recarregue a página.")

st.subheader("📊 Taxas de Acerto")
total_prev = len([h for h in st.session_state.historico if h["number"] not in (None, 0)]) - min_sorteios_para_prever
if total_prev > 0:
    acertos = len(st.session_state.acertos)
    taxa = acertos / total_prev * 100
    col_t = st.session_state.colunas_acertadas / total_prev * 100
    lin_t = st.session_state.linhas_acertadas / total_prev * 100
    st.info(f"🎯 Números: {taxa:.2f}% | 🧱 Colunas: {col_t:.2f}% | 📐 Linhas: {lin_t:.2f}%")
else:
    st.warning("Taxas serão exibidas após mais sorteios.")
