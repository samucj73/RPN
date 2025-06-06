import streamlit as st
from data_handler import fetch_latest_result, salvar_resultado_em_arquivo
from modelo_ia import prever_proximos_numeros_com_ia
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Monitor XXXtreme", layout="centered")
st.markdown("<h1 style='text-align:center;'>🎰 Monitor de Sorteios - XXXtreme Lightning Roulette</h1>", unsafe_allow_html=True)

# Atualiza a cada 10 segundos
st_autorefresh(interval=10_000, key="refresh")

MAX_HISTORY = 500

# Inicializa estados
if "history" not in st.session_state:
    st.session_state.history = []
if "last_seen_timestamp" not in st.session_state:
    st.session_state.last_seen_timestamp = None
if "ultima_previsao" not in st.session_state:
    st.session_state.ultima_previsao = None
if "prever_ativo" not in st.session_state:
    st.session_state.prever_ativo = False
if "acertos" not in st.session_state:
    st.session_state.acertos = []

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🚀 Iniciar Previsão IA" if not st.session_state.prever_ativo else "🛑 Parar Previsão"):
        st.session_state.prever_ativo = not st.session_state.prever_ativo

# Captura novo resultado
result = fetch_latest_result()

if result and result.get("timestamp") != st.session_state.last_seen_timestamp and result.get("number") is not None:
    st.session_state.history.insert(0, result)
    st.session_state.last_seen_timestamp = result.get("timestamp")

    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[:MAX_HISTORY]

    salvar_resultado_em_arquivo(result)

    # 🔮 Gera previsão automática se ativado
    if st.session_state.prever_ativo:
        with st.spinner("Gerando previsão com IA..."):
            previsoes_rapidas = prever_proximos_numeros_com_ia("resultados.csv", qtd=1)
        if previsoes_rapidas:
            st.session_state.ultima_previsao = previsoes_rapidas[0]
            if str(previsoes_rapidas[0]["numero"]) == str(result.get("number")):
                st.session_state.acertos.append({
                    "numero": previsoes_rapidas[0]["numero"],
                    "cor": previsoes_rapidas[0].get("cor", "N/A"),
                    "coluna": previsoes_rapidas[0].get("coluna", "N/A"),
                    "linha": previsoes_rapidas[0].get("linha", "N/A"),
                    "timestamp": result.get("timestamp")
                })

# Abas
abas = st.tabs(["📡 Monitoramento", "📈 Análise", "🔮 Previsões Futuras"])

with abas[0]:
    st.subheader("🎲 Números Sorteados ao Vivo")
    if st.session_state.history:
        for item in st.session_state.history[:10]:
            st.write(f"🎯 Número: {item['number']} | ⚡ Lucky: {item['lucky_numbers']} | 🕒 {item['timestamp']}")
    else:
        st.info("⏳ Aguardando os primeiros números...")

    st.markdown(f"📊 Números coletados: **{len(st.session_state.history)}**")

    if st.session_state.ultima_previsao:
        st.markdown("---")
        st.subheader("🔮 Última Previsão (IA):")
        prev = st.session_state.ultima_previsao
        st.markdown(
            f"🎯 **Número:** `{prev['numero']}` | 🎨 Cor: `{prev['cor']}` | 📊 Coluna: `{prev['coluna']}` | 🧱 Linha: `{prev['linha']}`"
        )

    if st.session_state.acertos:
        st.markdown("## ✅ Acertos da IA:")
        for acerto in st.session_state.acertos[-10:]:
            st.success(f"🎯 Acertou **{acerto['numero']}** em {acerto['timestamp']} (Cor: {acerto['cor']}, Coluna: {acerto['coluna']}, Linha: {acerto['linha']})")

with abas[1]:
    st.subheader("📊 Estatísticas dos Últimos Sorteios")
    if len(st.session_state.history) >= 10:
        if st.button("🔍 Analisar"):
            numeros = [item['number'] for item in st.session_state.history if item['number'] is not None]
            freq = {n: numeros.count(n) for n in set(numeros)}
            top_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            st.write("🎯 **Top 10 Números Mais Frequentes**:")
            for n, f in top_freq:
                st.write(f"➡️ Número {n} saiu {f} vezes")
    else:
        st.info("⏳ Aguardando mais dados para análise...")

with abas[2]:
    st.subheader("🔮 Previsão dos Próximos Números (IA)")
    with st.spinner("Gerando previsões com IA..."):
        previsoes = prever_proximos_numeros_com_ia("resultados.csv", qtd=10)
    if previsoes:
        for i, item in enumerate(previsoes, 1):
            texto = (
                f"**#{i}** 🎯 Número: `{item['numero']}` | 🎨 Cor: `{item['cor']}`"
                f" | 📊 Coluna: `{item['coluna']}` | 🧱 Linha: `{item['linha']}`"
                f" | ⬆⬇ Tipo: `{item['range']}` | 🔚 Terminal: `{item['terminal']}`"
                f" | ◀️ Vizinho Anterior: `{item['vizinho_anterior']}` | ▶️ Vizinho Posterior: `{item['vizinho_posterior']}`"
            )
            st.markdown(texto)
    else:
        st.warning("⚠️ Aguarde pelo menos 10 sorteios salvos para previsão com IA.")

st.markdown("<hr><p style='text-align:center'>© 2025 - Projeto de Previsão de Roleta com IA</p>", unsafe_allow_html=True)
