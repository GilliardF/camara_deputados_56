import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from data.distribuicao_uf import distribuicao_uf
from data.partidos_grouped import partidos_grouped

st.set_page_config(page_title="Partido Eleitoral", layout="wide", page_icon="📇")

db_path = os.path.abspath("data/data.db")

# Consultando siglas dos partidos
with sqlite3.connect(db_path) as conn:
    lista_partidos = pd.read_sql(
        """
        SELECT sigla
        FROM partidos
        """,
        conn,
    )

# Sidebar Partido Eleitoral
partido = sorted(lista_partidos["sigla"].unique().tolist())
select_partido = st.sidebar.selectbox("Sigla - Partido Eleitoral", partido)

despesas_partido = distribuicao_uf(select_partido, db_path)
partido_agrupado = partidos_grouped(select_partido, db_path)

# Dicionário de coordenadas (latitude, longitude) para cada UF
uf_coordinates = {
    "AC": (-9.0238, -70.8120),
    "AL": (-9.5713, -36.7820),
    "AM": (-3.4168, -65.8561),
    "AP": (0.9020, -52.0036),
    "BA": (-12.5797, -41.7007),
    "CE": (-5.4984, -39.3206),
    "DF": (-15.7801, -47.9292),
    "ES": (-19.1834, -40.3089),
    "GO": (-15.8270, -49.8362),
    "MA": (-4.9609, -45.2744),
    "MT": (-12.6819, -56.9211),
    "MS": (-20.7722, -54.7852),
    "MG": (-18.5122, -44.5550),
    "PA": (-3.4168, -52.2183),
    "PB": (-7.2400, -36.7820),
    "PR": (-25.2521, -52.0215),
    "PE": (-8.8137, -36.9541),
    "PI": (-7.7183, -42.7289),
    "RJ": (-22.9068, -43.1729),
    "RN": (-5.7945, -36.9541),
    "RS": (-30.0346, -51.2177),
    "RO": (-10.8305, -63.3333),
    "RR": (2.7376, -62.0751),
    "SC": (-27.2423, -50.2189),
    "SP": (-23.5505, -46.6333),
    "SE": (-10.9472, -37.0731),
    "TO": (-10.1833, -48.3333),
}

# Adicionando coordenadas ao DataFrame
partido_agrupado["latitude"] = partido_agrupado["siglaUf"].map(
    lambda uf: uf_coordinates[uf][0]
)
partido_agrupado["longitude"] = partido_agrupado["siglaUf"].map(
    lambda uf: uf_coordinates[uf][1]
)
partido_agrupado["Info"] = (
    partido_agrupado["siglaUf"]
    + " - "
    + partido_agrupado["total_deputados"].astype(str)
)

st.title(f"{select_partido}")

st.markdown("### 🌎 Distribuição de Deputados - UF")

total_deputados = partido_agrupado["total_deputados"].sum()
total_estados = partido_agrupado["siglaUf"].nunique()

with st.container():
    col1_tab1, col2_tab1 = st.columns([1, 3])
    with col1_tab1:
        st.metric("Total de Deputados", total_deputados)

    with col2_tab1:
        st.metric("Total de Estados", total_estados)

    fig1 = px.scatter_mapbox(
        partido_agrupado,
        lat="latitude",
        lon="longitude",
        size="total_deputados",
        color="total_deputados",
        zoom=3.4,
        labels={"siglaUf": "UF", "total_deputados": "Total de Deputados"},
        text="Info",
        mapbox_style="open-street-map",
        color_continuous_scale=px.colors.sequential.Sunset_r,
        center={"lat": -14.7017, "lon": -54.1253},
        hover_data={
            "latitude": False,
            "longitude": False,
            "Info": False,
            "total_deputados": False,
        },
        width=700,
        height=585,
    )

    fig1.update_traces(
        textfont=dict(size=12, color="black"),
        textposition="top center",
    )

    fig1.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    st.plotly_chart(fig1, use_container_width=True)
