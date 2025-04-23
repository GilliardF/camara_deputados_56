import streamlit as st
import plotly.express as px
import pandas as pd
import sqlite3
import os


db_path = "/home/gilliard/Documentos/camara_deputados_BI/datasets/data.db"


if not os.path.exists(db_path):
    st.error(f"🚨 Banco de dados não encontrado: {db_path}")
    st.stop()



st.title("Top 10 - Deputados com mais despesas")
with sqlite3.connect(db_path) as conn:
    df_top10_despesas = pd.read_sql_query(
        """SELECT deputados.id, deputados.urlFoto,
            deputados.nome, deputados.siglaUf,
            deputados.siglaPartido, 
            SUM(despesas.valorDocumento) AS totalDespesas
            FROM despesas
            INNER JOIN deputados ON despesas.id_deputado = deputados.id
            GROUP BY deputados.id, deputados.nome, deputados.urlFoto
            ORDER BY totalDespesas DESC
            LIMIT 10
        """, conn
    )



st.dataframe(
    df_top10_despesas,
    column_config={
        "id": st.column_config.NumberColumn(
            label="ID",
            format="%.0f"
        ),
        "urlFoto": st.column_config.ImageColumn(
            label="Foto",
            help="Imagem do deputado"
        ),
        "nome": "Nome do Deputado",
        "siglaUf": "UF",
        "siglaPartido": "Partido",
        "totalDespesas": st.column_config.NumberColumn(
            label="Total de Despesas",
            format="R$ %.0f"
        )
    },
    width=2000,
    hide_index=True
)


st.title("Top 10 - Classificação de Despesas")
with sqlite3.connect(db_path) as conn:
    df_class10_despesas = pd.read_sql_query(
        """SELECT despesas.tipoDespesa,
            SUM(despesas.valorDocumento) AS sumDespesas
            FROM despesas
            GROUP BY despesas.tipoDespesa
            ORDER BY sumDespesas ASC
            LIMIT 10
        """, conn
    )


fig = px.bar(
    df_class10_despesas,
    x="sumDespesas",
    y="tipoDespesa",
    orientation="h",
    title="Top 10 - Classificação de Despesas",
    labels={
        "sumDespesas": "Total de Despesas",
        "tipoDespesa": "Classificação de Despesas"
    },
    color_discrete_sequence=px.colors.qualitative.Prism
)


st.plotly_chart(fig, use_container_width=True)


st.title("Divisão de Despesas por UF")
with sqlite3.connect(db_path) as conn:
    df_uf_despesas = pd.read_sql_query(
        """SELECT deputados.siglaUf,
            SUM(despesas.valorDocumento) AS sumDespesas
            FROM despesas
            INNER JOIN deputados ON despesas.id_deputado = deputados.id
            GROUP BY deputados.siglaUf
            ORDER BY sumDespesas DESC
        """, conn
    )


fig = px.bar(
    df_uf_despesas,
    x="siglaUf",
    y="sumDespesas",
    title="Divisão de Despesas por UF",
    labels={
        "sumDespesas": "Total de Despesas",
        "siglaUf": "UF"
    },
    color_discrete_sequence=px.colors.qualitative.Prism
)

st.plotly_chart(fig, use_container_width=True)


st.title("Divisão de Despesas por Partido")
with sqlite3.connect(db_path) as conn:
    df_partido_despesas = pd.read_sql_query(
        """SELECT deputados.siglaPartido,
            SUM(despesas.valorDocumento) AS sumDespesas
            FROM despesas
            INNER JOIN deputados ON despesas.id_deputado = deputados.id
            GROUP BY deputados.siglaPartido
            ORDER BY sumDespesas DESC
        """, conn
    )


fig = px.bar(
    df_partido_despesas,
    x="siglaPartido",
    y="sumDespesas",
    title="Divisão de Despesas por Partido",
    labels={
        "sumDespesas": "Total de Despesas",
        "siglaPartido": "Partido"
    },
    color_discrete_sequence=px.colors.qualitative.Prism
)

st.plotly_chart(fig, use_container_width=True)


st.title("Divisão de Despesas por Mês")
with sqlite3.connect(db_path) as conn:
    df_mes_despesas = pd.read_sql_query(
        """SELECT strftime('%m', despesas.dataDocumento) AS mes,
            SUM(despesas.valorDocumento) AS sumDespesas
            FROM despesas
            GROUP BY mes
        """, conn
    )
    

fig = px.bar(
    df_mes_despesas,
    x="mes",
    y="sumDespesas",
    title="Divisão de Despesas por Mês",
    labels={
        "sumDespesas": "Total de Despesas",
        "mes": "Mês"
    },
    color_discrete_sequence=px.colors.qualitative.Prism
)

st.plotly_chart(fig, use_container_width=True)


