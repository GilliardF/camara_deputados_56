import webbrowser
import streamlit as st


st.set_page_config(
    page_title="Home",
    layout="wide",
    page_icon="🏠"
)

st.markdown("# Projeto Integrador Big Data II 🌐")
st.sidebar.markdown(
    "Desenvolvido por [Gilliard Fernandes](https://www.linkedin.com/in/gilliard-fernandes-bdata-dados/)"
)


btn1 = st.button("Acesse o repositório no GitHub")
if btn1:
    webbrowser.open_new_tab("")

st.write(
    """Este projeto busca coletar, armazenar e analisar dados públicos da Câmara dos Deputados 
    para entender os padrões de gastos parlamentares. Utilizando Big Data, 
    Business Intelligence (BI) e Machine Learning, serão gerados relatórios e visualizações interativas 
    que ajudarão a identificar tendências e possíveis irregularidades. O objetivo é transformar dados brutos 
    em informações valiosas, promovendo mais transparência e fiscalização dos recursos públicos.
    """
)
