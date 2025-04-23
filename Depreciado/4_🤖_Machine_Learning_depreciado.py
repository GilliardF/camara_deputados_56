import streamlit as st
import plotly.express as px
import pandas as pd
import sqlite3
import os

# Caminho absoluto do banco de dados
db_path = os.path.abspath("data/data.db")
conn = sqlite3.connect(db_path)

df_deputados = pd.read_sql_query("SELECT * FROM deputados_56", conn)
df_despesas = pd.read_sql_query("SELECT * FROM despesas", conn)
df_detalhes = pd.read_sql_query("SELECT * FROM deputados_56_detalhes", conn)
df_partidos = pd.read_sql_query("SELECT * FROM partidos", conn)

st.title("Dashboard de Análises da 56ª Legislatura")

# Adicionar componentes interativos
estado_selecionado = st.selectbox("Selecione um Estado", df_deputados["siglaUf"].unique())

# Filtrar dados pelo estado escolhido
df_filtrado = df_deputados[df_deputados["siglaUf"] == estado_selecionado]
st.write(df_filtrado)

df_merged = df_despesas.merge(df_deputados, left_on="id_deputado", right_on="id")
df_grouped = df_merged.groupby("siglaPartido")["valorLiquido"].sum().reset_index()

fig = px.bar(df_grouped, x="siglaPartido", y="valorLiquido", title="Total de Gastos por Partido")
st.plotly_chart(fig)

df_temporal = df_despesas.groupby(["ano", "mes"])["valorLiquido"].sum().reset_index()
fig = px.line(df_temporal, x="mes", y="valorLiquido", color="ano", title="Gastos Totais ao longo dos meses")
st.plotly_chart(fig)
