import webbrowser

import streamlit as st

st.set_page_config(page_title="Home", layout="wide", page_icon="🏠")

st.markdown("# Projeto Integrador Big Data II 🌐")
st.sidebar.markdown(
    "Desenvolvido por [Gilliard Fernandes](https://www.linkedin.com/in/gilliard-fernandes-bdata-dados/)"
)

btn1 = st.button("Acesse o repositório no GitHub")
if btn1:
    webbrowser.open_new_tab("https://github.com/GilliardF/camara_deputados_56")

st.write(
    """
    O projeto visa coletar, armazenar e analisar dados públicos referente aos
    gastos de 2022 dos deputados da 56° legislatura, utilizando
    tecnologias de Big Data, Business Intelligence (BI) e Machine Learning
    para transformar informações brutas em insights valiosos.

    Através do Streamlit, geraremos relatórios e visualizações
    interativas que facilitam a compreensão dos padrões de gastos,
    enquanto algoritmos de Machine Learning, como a Análise de
    Componentes Principais (PCA), serão aplicados para a detecção
    inteligente de anomalias.

    Especificamente, o PCA nos ajudará a identificar gastos potencialmente
    irregulares ao analisar a variância nos valores das despesas – variações
    muito grandes ou atípicas podem ser sinalizadas automaticamente
    como anomalias pelo modelo.
    """
)
