import streamlit as st
from data_handler import fetch_latest_result, salvar_resultado_em_arquivo
from modelo_ia import prever_proximos_numeros_com_ia
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Monitor XXXtreme", layout="centered")
st.markdown("<h1 style='text-align:center;'>🎰 Monitor de Sorteios - XXXtreme Lightning Roulette</h1>", unsafe_allow_html=True)

# Auto refresh a cada 10 segundos
st_autorefresh(interval=10_000, key="refresh")

# Estado da sessão
if "history" not in st.session_state:
    st.session_state.history = []
if "last_seen_timestamp" not in st.session_state:
    st.session_state.last_seen_timestamp = None
if "ultima_previsao" not in st.session_state:
    st.session_state.ultima_previsao = None
if "acerto_ia" not in st.session_state:
    st.session_state.acerto_ia = False

# Captura novo sorteio
result = fetch_latest_result()
if result and result.get("timestamp") != st.session_state.last_seen_timestamp:
    st.session_state.history.insert(0, result)
    st.session_state.last_seen_timestamp = result.get("timestamp")
    salvar_resultado_em_arquivo(result)

    # Gera previsão imediata
    previsoes_rapidas = prever_proximos_numeros_com_ia("resultados.csv", qtd=1)
    if previsoes_rapidas:
        st.session_state.ultima_previsao = previsoes_rapidas[0]

    # Verifica acerto da previsão anterior
    ultima_previsao = st.session_state.ultima_previsao
    numero_sorteado = result.get("number") or result.get("numero")
    if ultima_previsao and numero_sorteado == ultima_previsao["numero"]:
        st.session_state.acerto_ia = True
    else:
        st.session_state.acerto_ia = False

# --- TABS ---
abas = st.tabs(["📡 Monitoramento", "📈 Análise", "🔮 Previsões Futuras"])

# 🟠 Aba 1 – Monitoramento
with abas[0]:
    st.subheader("🎲 Números Sorteados ao Vivo")

    if st.session_state.history:
        for item in st.session_state.history[:10]:
            numero = item.get("number") or item.get("numero", "N/A")
            lucky = item.get("lucky_numbers") or item.get("lucky", "N/A")
            timestamp = item.get("timestamp", "N/A")
            st.write(f"🎯 Número: {numero} | ⚡ Lucky: {lucky} | 🕒 {timestamp}")
    else:
        st.info("⏳ Aguardando os primeiros números...")

    st.markdown(f"📊 Números coletados: **{len(st.session_state.history)}**")

    if st.session_state.ultima_previsao:
        st.markdown("---")
        st.subheader("🔮 Próximo Número Previsto (IA em tempo real):")
        prev = st.session_state.ultima_previsao

        if st.session_state.acerto_ia:
            st.success(
                f"🎯 **ACERTO!** A previsão da IA estava certa: `{prev['numero']}`"
            )
            st.markdown(
                """
                <audio autoplay>
                    <source src="https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg" type="audio/ogg">
                </audio>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"🎯 **Número:** `{prev['numero']}` | 🎨 Cor: `{prev['cor']}` | 📊 Coluna: `{prev['coluna']}` | 🧱 Linha: `{prev['linha']}`"
            )

# 🟡 Aba 2 – Análise
with abas[1]:
    st.subheader("📊 Estatísticas dos Últimos Sorteios")
    if len(st.session_state.history) >= 10:
        if st.button("🔍 Analisar"):
            numeros = [
                item.get("number") or item.get("numero")
                for item in st.session_state.history if (item.get("number") or item.get("numero")) is not None
            ]
            freq = {n: numeros.count(n) for n in set(numeros)}
            top_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]

            st.write("🎯 **Top 10 Números Mais Frequentes**:")
            for n, f in top_freq:
                st.write(f"➡️ Número {n} saiu {f} vezes")

# 🟢 Aba 3 – Previsões Futuras (IA)
with abas[2]:
    st.subheader("🔮 Previsão dos Próximos Números (IA)")

    previsoes = prever_proximos_numeros_com_ia("resultados.csv", qtd=10)

    if previsoes:
        numeros_sorteados = [
            item.get("number") or item.get("numero")
            for item in st.session_state.history[:1]
        ]

        for i, item in enumerate(previsoes, 1):
            texto = (
                f"**#{i}** 🎯 Número: `{item['numero']}` | 🎨 Cor: `{item['cor']}`"
                f" | 📊 Coluna: `{item['coluna']}` | 🧱 Linha: `{item['linha']}`"
                f" | ⬆⬇ Tipo: `{item['range']}` | 🔚 Terminal: `{item['terminal']}`"
                f" | ◀️ Vizinho Anterior: `{item['vizinho_anterior']}` | ▶️ Vizinho Posterior: `{item['vizinho_posterior']}`"
            )
            if item['numero'] in numeros_sorteados:
                st.success(texto)
            else:
                st.markdown(texto)
    else:
        st.info("🔄 Aguarde mais dados (mínimo 30 sorteios) para previsão com IA.")

# Rodapé padrão
st.markdown("<hr><p style='text-align:center'>© 2025 - Projeto de Previsão de Roleta com IA</p>", unsafe_allow_html=True)
