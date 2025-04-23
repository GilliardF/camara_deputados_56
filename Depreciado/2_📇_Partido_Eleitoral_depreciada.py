import streamlit as st
import plotly.express as px
import pandas as pd
import sqlite3
import os


st.set_page_config(
    page_title="Análise de Partidos Eleitorais",
    layout="wide",
    page_icon="📇"
)

db_path = os.path.abspath("datasets/data.db")

# Deputados por Partido - UF
with sqlite3.connect(db_path) as conn:
    deputados_partido_uf = pd.read_sql_query(
        """SELECT
                partidos.sigla,
                partidos.nome,
                deputados_56.siglaUf,
                COUNT(deputados_56.id) AS total_deputados
            FROM
                partidos
            INNER JOIN
                deputados_56 ON partidos.sigla = deputados_56.siglaPartido
            GROUP BY
                partidos.sigla, partidos.nome, deputados_56.siglaUf
            ORDER BY
                total_deputados DESC;
        """, conn
    ) # Tabela usada na 1° Aba - fig1


# Sidebar Partido Eleitoral
partido = sorted(deputados_partido_uf["sigla"].unique().tolist())
select_partido = st.sidebar.selectbox("Sigla - Partido Eleitoral", partido)

# Ranking de Gastos por Deputado
with sqlite3.connect(db_path) as conn:
    ranking_gastos = pd.read_sql_query(
        """
        SELECT
            deputados_56.siglaUf,
            SUM(despesas.valorDocumento) AS total_gasto
        FROM
            deputados_56
        INNER JOIN
            partidos ON deputados_56.siglaPartido = partidos.sigla
        INNER JOIN
            despesas ON deputados_56.id = despesas.id_deputado
        WHERE
            siglaPartido = ?
        GROUP BY
            deputados_56.siglaUf
        ORDER BY total_gasto DESC;
        """, conn, params=(select_partido,) # Tabela usada na 2° Aba - DataFrame
    )


with sqlite3.connect(db_path) as conn:
    lista_uf_partidos = pd.read_sql_query(
        """
        SELECT
            deputados_56.siglaUf,
            deputados_56.siglaPartido
        FROM
            deputados_56
        WHERE
            deputados_56.siglaPartido = ?
        GROUP BY
            deputados_56.siglaUf;
        """, conn, params=(select_partido,)
    ) # Usada em st.metrics na 1° Aba


# Despesas por Partido
with sqlite3.connect(db_path) as conn:
    despesas_partidos = pd.read_sql_query(
        """SELECT
                deputados_56.siglaPartido,
                deputados_56.siglaUf,
                despesas.tipoDespesa,
                SUM(despesas.valorDocumento) AS total_gasto
            FROM
                deputados_56
            INNER JOIN
                despesas ON deputados_56.id = despesas.id_deputado
            INNER JOIN
                partidos ON deputados_56.siglaPartido = partidos.sigla
            WHERE
                deputados_56.siglaPartido = ?
            GROUP BY
                deputados_56.siglaPartido,
                despesas.tipoDespesa
            ORDER BY
                total_gasto ASC; 
        """, conn, params=(select_partido,) # Tabela usada na 2° Aba - fig2
    )

# Despesas por Partido durante o Ano
with sqlite3.connect(db_path) as conn:
    despesas_partido_ano = pd.read_sql_query(
        """
        SELECT
            deputados_56.siglaPartido,
            despesas.mes,
            deputados_56.siglaUf,
            SUM(despesas.valorDocumento) AS total_gasto
        FROM
            deputados_56
        INNER JOIN
            despesas ON deputados_56.id = despesas.id_deputado
        WHERE
            deputados_56.siglaPartido = ?
        GROUP BY
            deputados_56.siglaPartido, deputados_56.siglaUf, despesas.mes
        ORDER BY
            total_gasto ASC;
        """, conn, params=(select_partido,) # Tabela usada na 2° Aba - fig3
    )


# Gastos por Fornecedor
with sqlite3.connect(db_path) as conn:
    gastos_fornecedor = pd.read_sql_query(
        """
        SELECT
            despesas.nomeFornecedor,
            SUM(despesas.valorDocumento) AS total_gasto
        FROM
            deputados_56
        INNER JOIN
            despesas ON deputados_56.id = despesas.id_deputado
        WHERE
            deputados_56.siglaPartido = ?
        GROUP BY
            deputados_56.siglaPartido, despesas.nomeFornecedor
        ORDER BY
            total_gasto DESC;
        """, conn, params=(select_partido,) # Tabela usada na 3° Aba - DataFrame
    )


# Fornecedores por UF
with sqlite3.connect(db_path) as conn:
    fornecedores_uf = pd.read_sql_query(
        """
        SELECT
            deputados_56.siglaUf,
            despesas.nomeFornecedor,
            SUM(despesas.valorDocumento) AS total_gasto
        FROM
            deputados_56
        INNER JOIN
            despesas ON deputados_56.id = despesas.id_deputado
        WHERE
            deputados_56.siglaPartido = ?
        GROUP BY
            deputados_56.siglaUf, despesas.nomeFornecedor
        ORDER BY
            total_gasto DESC;
        """, conn, params=(select_partido,) # Tabela usada na 3° Aba - DataFrame
    )


# Filtrar DF deputados_partido_uf pelo Sigla Partido
deputados_partido_uf = deputados_partido_uf[deputados_partido_uf["sigla"] == select_partido]
nome_partido = deputados_partido_uf["nome"][deputados_partido_uf["sigla"] == select_partido].values[0]
total_deputados_partido = deputados_partido_uf["total_deputados"].sum()


st.title(f"{select_partido} - {nome_partido}")

tab1, tab2, tab3 = st.tabs([
    "🌎 Distribuição de Deputados - UF",
    "💰 Despesas",
    "🏬 Fornecedores"
])


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
    "TO": (-10.1833, -48.3333)
}


# Adicionar coordenadas de latitude e longitude ao DataFrame
deputados_partido_uf = deputados_partido_uf[deputados_partido_uf["sigla"] == select_partido]
deputados_partido_uf["Latitude"] = deputados_partido_uf["siglaUf"].map(lambda x: uf_coordinates.get(x, (None, None))[0])
deputados_partido_uf["Longitude"] = deputados_partido_uf["siglaUf"].map(lambda x: uf_coordinates.get(x, (None, None))[1])

# Criar coluna de texto informativo
deputados_partido_uf["Info"] = deputados_partido_uf["siglaUf"] + ": " + deputados_partido_uf["total_deputados"].astype(str)
  

with tab1:
    col1_tab1, col2_tab1 = st.columns([1, 3])
    with col1_tab1:
        st.markdown("###### Média de Gastos por UF")

        # Calcula a média dos gastos por UF
        media_gastos_por_uf = ranking_gastos.groupby("siglaUf")["total_gasto"].mean().to_dict()

        # Itera sobre as UFs presentes no partido selecionado
        for uf in lista_uf_partidos["siglaUf"]:
            # Verifica se a UF tem registros de gastos
            if uf in media_gastos_por_uf:
                valor_exibir = f"R$ {media_gastos_por_uf[uf]:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            else:
                valor_exibir = "R$ 0,00"

            # Exibe a métrica no Streamlit
            st.metric(label=uf, value=valor_exibir)


    with col2_tab1:
        col1_1_tab1, col2_1_tab1 = st.columns([1, 1])

        with col1_1_tab1:
            st.metric("Total de Estados Presentes", deputados_partido_uf["siglaUf"].nunique())

        with col2_1_tab1:
            st.metric("Total de Deputados", total_deputados_partido)

        fig1 = px.scatter_map(
            deputados_partido_uf,
            lat="Latitude",
            lon="Longitude",
            size="total_deputados",
            color="total_deputados",
            zoom=3.4,
            labels={
                "siglaUf": "UF",
                "total_deputados": "Total de Deputados"
            },
            text="Info",
            map_style="open-street-map",
            color_continuous_scale=px.colors.sequential.Sunset_r,
            center={"lat": -14.7017, "lon": -54.1253},
            hover_data={"Latitude": False, "Longitude": False, "Info": False, "total_deputados": False},
            width=700,
            height=585
        )

        fig1.update_traces(
            textfont=dict(size=12, color="black"),
            textposition="top center"
        )

        fig1.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

        st.plotly_chart(fig1, use_container_width=True)


with tab2:
    col1_tab2, col2_tab2 = st.columns([1, 2])
    with col1_tab2:
        st.markdown(f"###### Ranking de Gastos por Deputado")
        st.dataframe(
            ranking_gastos.sort_values(by='siglaUf'),
            height=500,
            hide_index=True,
            column_config={
                "total_gasto": st.column_config.ProgressColumn(
                    "total_gasto", format="R$%.2f",
                    min_value=0, max_value=ranking_gastos["total_gasto"].max()
                ),
                "urlFoto": st.column_config.ImageColumn("urlFoto")
            }
        )


    with col2_tab2:        
        st.markdown(f"###### Classificação de Despesas por Partido")
        fig2 = px.bar(
            despesas_partidos,
            x="total_gasto",
            y="tipoDespesa",
            labels={
                "total_gasto": "Total de Despesas"
            },
            color_discrete_sequence=px.colors.qualitative.Bold,
            orientation='h',
            width=800,
            height=600,
            text_auto='.2s'
        )

        fig2.update_layout(
            xaxis=dict(
                showticklabels=False
            ),
            yaxis=dict(
                tickfont=dict(size=12, color='black', family='Arial', weight='bold')
            )
        )

        fig2.update_traces(
            texttemplate='%{x:.2s}',
            textposition='outside',
            textfont=dict(
                size=12,
                color='black',
                family='Arial',
                weight='bold',
            )
        )

        st.plotly_chart(fig2, use_container_width=True)

    with st.container():
        st.markdown("###### Despesas por UF Distribuídas por Mês")
        fig3 = px.bar(
            despesas_partido_ano[despesas_partido_ano["siglaPartido"] == select_partido],
            x="mes",
            y="total_gasto",
            color="siglaUf",
            labels={
                "mes": "Mês",
                "total_gasto": "Total de Despesas",
                "siglaUf": "UF"
            },
            width=800,
            height=600
        )
        
        fig3.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                ticktext=["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
            )
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
with tab3:
    with st.container():
        # Ordenar os fornecedores pelo total gasto em ordem decrescente
        partido_fornecedor = gastos_fornecedor.groupby('nomeFornecedor')['total_gasto'].sum().reset_index()
        partido_fornecedor = partido_fornecedor.sort_values(by='total_gasto', ascending=False)

        # Selecionar fornecedores únicos
        fornecedores = st.multiselect("Selecione um ou vários Fornecedor(es)", partido_fornecedor["nomeFornecedor"].unique())

        # Filtrar os dados de acordo com os fornecedores selecionados
        if fornecedores:
            select_fornecedores = fornecedores_uf[fornecedores_uf["nomeFornecedor"].isin(fornecedores)]
        else:
            select_fornecedores = fornecedores_uf.head(10)

    col1_tab3, col2_tab3 = st.columns([1, 2])
    with col1_tab3:
        st.markdown(f"###### Ranking de Gastos por Fornecedor")
        st.dataframe(
            gastos_fornecedor,
            height=500,
            hide_index=True,
            column_config={
                "total_gasto": st.column_config.ProgressColumn(
                    "total_gasto", format="R$%.2f",
                    min_value=0, max_value=gastos_fornecedor["total_gasto"].max()
                )
            }
        )

    with col2_tab3:
        st.markdown(f"###### Fornecedores por UF")
        fig4 = px.bar(
            select_fornecedores,
            x="total_gasto",
            y="nomeFornecedor",
            color="siglaUf",
            labels={
                "total_gasto": "Total de Gastos",
                "nomeFornecedor": "Fornecedor",
                "siglaUf": "UF"
            },
            orientation='h',
            width=800,
            height=600
        )

        fig4.update_layout(
            xaxis=dict(
                showticklabels=False
            ),
            yaxis=dict(
                tickfont=dict(size=12, color='black', family='Arial', weight='bold')
            )
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    