import streamlit as st
import plotly.express as px
import pandas as pd
import sqlite3
import os

# Caminho absoluto do banco de dados
db_path = os.path.abspath("data/data.db")
conn = sqlite3.connect(db_path)

# Configuração da página
st.set_page_config(
    page_title="Machine Learning",
    layout="wide",
    page_icon="🤖"
)


# Consultas para obter listas necessárias para o(s) filtro(s)
with sqlite3.connect(db_path) as conn:
    # Lista de partidos únicos
    lista_partidos = pd.read_sql(
        "SELECT DISTINCT siglaPartido FROM deputados_56 ORDER BY siglaPartido",
        conn
    )

    # Lista de combinações UF x Partido
    lista_uf = pd.read_sql(
        "SELECT DISTINCT siglaUf, siglaPartido FROM deputados_56",
        conn
    )

    # Lista completa de deputados
    lista_deputados = pd.read_sql(
        """
        SELECT
            nome,
            siglaPartido,
            siglaUf
        FROM deputados_56
        """, conn
    )

# Multiselect para escolher os partidos
partidos = sorted(lista_partidos["siglaPartido"].unique().tolist())
select_partidos = st.multiselect(
    "Partido Eleitoral",
    partidos,
    default=partidos
)

# Multiselect para escolher os estados
ufs = sorted(lista_uf["siglaUf"].unique().tolist())
select_ufs = st.multiselect(
    "Unidade Federativa",
    ufs,
    default=ufs
)

# Multiselect para escolher os deputados
deputados = sorted(lista_deputados["nome"].unique().tolist())
select_deputados = st.multiselect(
    "Deputado",
    deputados,
    default=deputados
)
