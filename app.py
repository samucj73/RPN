import streamlit as st
from data_handler import fetch_latest_result, salvar_resultado_em_arquivo
from modelo_ia import prever_proximos_numeros_com_ia
from streamlit_autorefresh import st_autorefresh
import pandas as pd

st.set_page_config(page_title="Monitor XXXtreme", layout="centered")
st.markdown("<h1 style='text-align:center;'>🎰 XXXtreme Lightning Roulette</h1>", unsafe_allow_html=True)

st_autorefresh(interval=10_000, key="refresh")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_seen_timestamp" not in st.session_state:
    st.session_state.last_seen_timestamp = None
if "ultima_previsao" not in st.session_state:
    st.session_state.ultima_previsao = None

result = fetch_latest_result()
if result and result.get("timestamp") != st.session_state.last_seen_timestamp:
    st.session_state.history.insert(0, result)
    st.session_state.history = st.session_state.history[:50]
    st.session_state.last_seen_timestamp = result.get("timestamp")
    salvar_resultado_em_arquivo(result)

    previsoes_rapidas = prever_proximos_numeros_com_ia("resultados.csv", qtd=1)
    if previsoes_rapidas:
        st.session_state.ultima_previsao = previsoes_rapidas[0]

abas = st.tabs(["📡 Monitor", "📊 Análise", "🔮 Previsão"])

with abas[0]:
    st.subheader("🎲 Últimos Sorteios")
    if st.session_state.history:
        for item in st.session_state.history[:10]:
            st.code(f"{item.get('timestamp', '')} | Nº: {item.get('number') or item.get('numero')} | Lucky: {item.get('lucky_numbers') or item.get('lucky')}")
        st.markdown(f"**Total Coletado:** {len(st.session_state.history)} / 50")
    else:
        st.info("Aguardando sorteios...")

    if st.session_state.ultima_previsao:
        st.markdown("---")
        prev = st.session_state.ultima_previsao
        st.subheader("🔮 Previsão Imediata (IA)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Número", prev['numero'])
        col2.metric("Cor", prev['cor'])
        col3.metric("Coluna", prev['coluna'])
        col4.metric("Linha", prev['linha'])

with abas[1]:
    st.subheader("📊 Frequência dos Últimos Números")
    if len(st.session_state.history) >= 10:
        if st.button("📈 Analisar Frequência"):
            numeros = [
                item.get("number") or item.get("numero")
                for item in st.session_state.history if (item.get("number") or item.get("numero")) is not None
            ]
            freq = {n: numeros.count(n) for n in set(numeros)}
            df_freq = pd.DataFrame(sorted(freq.items()), columns=["Número", "Frequência"]).sort_values(by="Frequência", ascending=False)

            st.write("🔝 Top Números:")
            st.dataframe(df_freq.head(10), use_container_width=True)
            st.write("📊 Gráfico de Frequência:")
            st.bar_chart(df_freq.set_index("Número"))
    else:
        st.info("É necessário ao menos 10 sorteios para análise.")

with abas[2]:
    st.subheader("🔮 Previsões Futuras (IA)")
    previsoes = prever_proximos_numeros_com_ia("resultados.csv", qtd=10)
    if previsoes:
        ultimos = [item.get("number") or item.get("numero") for item in st.session_state.history[:1]]
        for i, item in enumerate(previsoes, 1):
            texto = (
                f"{i:02d} ➤ Nº: `{item['numero']}` | Cor: `{item['cor']}` | Col: `{item['coluna']}` | Linha: `{item['linha']}` "
                f"| Tipo: `{item['range']}` | Terminal: `{item['terminal']}` | ◀️ `{item['vizinho_anterior']}` ▶️ `{item['vizinho_posterior']}`"
            )
            if item['numero'] in ultimos:
                st.success(texto)
            else:
                st.markdown(texto)
    else:
        st.info("🔄 Aguarde mais dados (mínimo 30 sorteios) para gerar previsões.")

st.markdown("<hr><p style='text-align:center;'>© 2025 - IA para Roleta XXXtreme</p>", unsafe_allow_html=True)
