import streamlit as st
import plotly.express as px
import pandas as pd
import sqlite3
import os


st.set_page_config(
    page_title="Análise Dep. Federal",
    layout="wide",
    page_icon="👤"
)

db_path = os.path.abspath("data/data.db")

with sqlite3.connect(db_path) as conn:
    df_deputados = pd.read_sql_query(
        """
        SELECT
            d56.id,
            d56.nome,
            d56_d.nomeCivil,
            d56_d.cpf,
            d56_d.dataNascimento,
            d56_d.dataFalecimento,
            d56_d.escolaridade,
            d56_d.profissoes,
            d56_d.redeSocial,
            d56.siglaUf,
            d56.siglaPartido,
            p.nome AS nome_partido,
            p.urlLogo AS logo_partido,
            d56.urlFoto
        FROM deputados_56 d56
        INNER JOIN deputados_56_detalhes d56_d ON d56.id = d56_d.id
        INNER JOIN partidos p ON d56.siglaPartido = p.sigla;
        """,
        conn
    )

# Sidebar UF
df_uf = sorted(df_deputados["siglaUf"].unique().tolist())
select_uf = st.sidebar.selectbox("UF", df_uf)

# Sidebar Partido Eleitoral
df_partido = df_deputados[df_deputados["siglaUf"] == select_uf]
partido = sorted(df_partido["siglaPartido"].unique().tolist())
select_partido = st.sidebar.selectbox("Sigla - Partido Eleitoral", partido)

# Sidebar deputados
df_deputado = df_deputados[
    (df_deputados["siglaUf"] == select_uf) &
    (df_deputados["siglaPartido"] == select_partido)
]

deputado = sorted(df_deputado["nome"].tolist())
select_deputado = st.sidebar.selectbox("Deputado(s)", deputado)

# Descrição do deputado
deputado_info = df_deputados[
    (df_deputados["siglaUf"] == select_uf) &
    (df_deputados["siglaPartido"] == select_partido) &
    (df_deputados["urlFoto"] == df_deputados["urlFoto"]) &
    (df_deputados["nome"] == select_deputado)
]

row1 = st.columns(3)


with st.container():
    row1[0].image(deputado_info['urlFoto'].values[-1], width=100)
    row1[1].image(deputado_info['logo_partido'].values[-1], width=50)


tab1, tab2, tab3 = st.tabs([
    "📔 Informações Gerais",
    "💰 Despesas",
    "🏬 Fornecedores"
])

with tab1:
    st.write(f"**Nome Civil**: {deputado_info['nomeCivil'].values[-1]}")
    st.write(f"**UF**: {deputado_info['siglaUf'].values[-1]}")
    st.write(f"**Partido**: {deputado_info['nome_partido'].values[-1]}")
    st.write(f"**CPF**: {deputado_info['cpf'].values[-1]}")
    st.write(f"**Data de Nascimento**: {deputado_info['dataNascimento'].values[-1]}")
    st.write(f"**Data de Falecimento**: {deputado_info['dataFalecimento'].values[-1]}")
    st.write(f"**Escolaridade**: {deputado_info['escolaridade'].values[-1]}")
    st.write(f"**Profissão**: {deputado_info['profissoes'].values[-1]}")

    for redes_sociais in deputado_info['redeSocial'].values:
        # Obtendo a lista de redes sociais a partir da string separada por '; '
        lista_rede_social = redes_sociais.split('; ')

        # Exibindo as redes sociais com links clicáveis, um por vez
        for each in lista_rede_social:
            if "facebook" in each:
                st.write(f"**Facebook**: [{each}]({each})")
            elif "twitter" in each:
                st.write(f"**Twitter**: [{each}]({each})")
            elif "instagram" in each:
                st.write(f"**Instagram**: [{each}]({each})")
            else:
                st.write(f"**Outros**: [{each}]({each})")


 
select_despesas_deputado = df_deputado_despesas[['tipoDespesa', 'valorDocumento']]
select_despesas_deputado.columns = ['Tipo de Despesa', 'Valor (R$)']
select_despesas_deputado = select_despesas_deputado.sort_values(by="Valor (R$)", ascending=False).reset_index(drop=True)

# Função para estilizar as 5 primeiras linhas
def top_5_tipo_despesa(row):
    if row.name < 5:  
        return ['background-color: #ffff7a; color: #000000; font-weight: bold'] * len(row)
    return [''] * len(row)

styled_df = select_despesas_deputado.style.apply(top_5_tipo_despesa, axis=1)
styled_df = styled_df.format({f"Valor (R$)": "R${:,.2f}"})

with sqlite3.connect(db_path) as conn:
    df_meses_despesas = pd.read_sql_query(
        """
        SELECT
            deputados_56.id,
            deputados_56.nome,
            strftime('%m', despesas.dataDocumento) AS mes,
            SUM(despesas.valorDocumento) AS valorDocumento
        FROM despesas
        INNER JOIN deputados_56 ON despesas.id_deputado = deputados_56.id
        WHERE deputados_56.nome = ?
        GROUP BY
            mes
        ORDER BY mes;
        """, conn, params=(select_deputado,)
    )

select_meses_deputado = df_meses_despesas[['mes', 'valorDocumento']]
select_meses_deputado.columns = ['Mês', 'Valor (R$)']
select_meses_deputado = select_meses_deputado.sort_values(by="Valor (R$)", ascending=False).reset_index(drop=True)

with tab2:
    st.dataframe(
        styled_df,
        hide_index=True
    )

    fig1 = px.bar(
        select_meses_deputado,
        x='Mês',
        y='Valor (R$)',
        title=f'Despesas por Mês de {select_deputado}',
        labels={'Valor (R$)': 'Valor (R$)'}
    )
    
    st.plotly_chart(fig1, key="fig1")


with sqlite3.connect(db_path) as conn:
    df_despesas_fornecedor = pd.read_sql_query(
        """
        SELECT
            deputados_56.id,
            deputados_56.nome,
            despesas.nomeFornecedor,
            SUM(despesas.valorDocumento) AS valorDocumento
        FROM despesas
        INNER JOIN deputados_56 ON despesas.id_deputado = deputados_56.id
        WHERE deputados_56.nome = ?  -- usamos um placeholder para prevenir SQL Injection
        GROUP BY
            despesas.nomeFornecedor
        ORDER BY despesas.nomeFornecedor;
        """, conn, params=(select_deputado,)
    )

select_fornecedor_deputado = df_despesas_fornecedor[['nomeFornecedor', 'valorDocumento']]
select_fornecedor_deputado.columns = ['Fornecedor', 'Valor (R$)']
select_fornecedor_deputado = select_fornecedor_deputado.sort_values(by="Valor (R$)", ascending=False).reset_index(drop=True)

with tab3:
    st.dataframe(
        select_fornecedor_deputado,
        hide_index=True
    )
    