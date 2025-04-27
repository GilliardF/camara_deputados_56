from data.deputado_detalhe import deputado_detalhe
from data.deputados_historico import deputados_historico
from data.deputado_despesas import despesas_deputado
import streamlit as st
import plotly.express as px
import pandas as pd
import sqlite3
import os

# Configuração da página
st.set_page_config(
    page_title="Deputado Federal",
    layout="wide",
    page_icon="👤"
)

# Localização do banco de dados
db_path = os.path.abspath("data/data.db")


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

# Sidebar Partido
partido = sorted(lista_partidos["siglaPartido"].unique().tolist())
select_partido = st.sidebar.selectbox("Sigla - Partido Eleitoral", partido)

# Sidebar Estado para o partido selecionado
df_estado = lista_uf[lista_uf["siglaPartido"] == select_partido]
estado = sorted(df_estado["siglaUf"].unique().tolist())
select_estado = st.sidebar.selectbox("UF", estado)

# Sidebar deputados
df_deputado = lista_deputados[
    (lista_deputados["siglaPartido"] == select_partido) &
    (lista_deputados["siglaUf"] == select_estado)
]

deputado = sorted(df_deputado["nome"].unique().tolist())
select_deputado = st.sidebar.selectbox("Deputado(s)", deputado)

deputado_info = deputado_detalhe(select_deputado, db_path)
id_deputado = deputado_info['id'].values[0]
deputado_historico = deputados_historico(int(id_deputado), db_path)
deputado_despesas = despesas_deputado(int(id_deputado), db_path)

row1 = st.columns(3)

with st.container():
    # Exibir a imagem do deputado
    row1[0].image(deputado_info['urlFoto'].values[-1], width=100)

    # Verifica e exibe a logo do partido correspondente ao partido selecionado
    # Obtém a logo do partido apenas se ela for correspondente ao partido selecionado
    partido_logo_info = deputado_info.loc[deputado_info['siglaPartido'] == select_partido, 'urlLogo']

    if not partido_logo_info.empty:
        logo_url = partido_logo_info.values[0]
        if pd.notna(logo_url) and logo_url.strip():
            row1[1].image(logo_url, width=100)

tab1, tab2, tab3 = st.tabs([
    "📔 Informações Gerais",
    "💰 Despesas",
    "🏬 Fornecedores"
])

# Função para formatar a data em dd/mm/aaaa e se for vazio retorna "Não informado"
def formatar_data(data):
    return data.strftime("%d/%m/%Y") if not pd.isna(data) else "Não informado"

data_nascimento = formatar_data(pd.to_datetime(deputado_info['dataNascimento'].values[-1]))
data_falecimento = formatar_data(pd.to_datetime(deputado_info['dataFalecimento'].values[-1]))

# Função para formatar o CPF
def formatar_cpf(cpf):
    return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}" if cpf else "Não informado"

cpf = formatar_cpf(deputado_info['cpf'].values[-1])

with tab1:
    col1_tab1, col2_tab1 = st.columns([2, 3])
    with col1_tab1:
        nome = deputado_info['nomeCivil'].values[-1]
        st.write(f"**Nome Civil**: {nome.title()}")
        st.write(f"**UF**: {deputado_info['siglaUf'].values[-1]}")
        st.write(f"**CPF**: {cpf}")
        st.write(f"**Data de Nascimento**: {data_nascimento}")
        st.write(f"**Data de Falecimento**: {data_falecimento}")
        st.write(f"**Escolaridade**: {deputado_info['escolaridade'].values[-1]}")
        st.write(f"**Profissão**: {deputado_info['profissoes'].values[-1]}")

        todas_redes_sociais = set()

        for redes_sociais in deputado_info['redeSocial'].values:
            for rede in redes_sociais.split(';'):
                rede_limpa = rede.strip()
                if rede_limpa:
                    todas_redes_sociais.add(rede_limpa)

        for each in sorted(todas_redes_sociais):
            if "facebook" in each:
                st.write(f"**Facebook**: [{each}]({each})")
            elif "twitter" in each:
                st.write(f"**Twitter**: [{each}]({each})")
            elif "instagram" in each:
                st.write(f"**Instagram**: [{each}]({each})")
            else:
                st.write(f"**Outros**: [{each}]({each})")

    with col2_tab1:
        col2_1_tab1, col2_2_tab1 = st.columns([1, 1])

        deputados_filtrados = lista_deputados[lista_deputados['nome'] == select_deputado]
        partidos = deputados_filtrados['siglaPartido'].unique().tolist()
        partidos = sorted(partidos)
        partidos_str = ", ".join(partidos) if partidos else "Nenhum outro partido"

        with col2_1_tab1:
            # Total de partidos
            st.metric(label="Total de partidos", value=len(partidos))

        with col2_2_tab1:
            # Exibe a lista de partidos
            st.metric(label="Partido(s) que participou durante 56° Legislatura", value=partidos_str)

        st.markdown("**Histórico do(s) Partido(s) em 2022:**")
        deputado_historico['dataHora'] = pd.to_datetime(deputado_historico['dataHora'])
        deputado_historico['dataHora'] = deputado_historico['dataHora'].dt.strftime('%d/%m/%Y %H:%M')

        # Renomeando as colunas
        deputado_historico = deputado_historico.rename(columns={
            'siglaPartido': 'Sigla - Partido',
            'dataHora': 'Data - Hora',
            'situacao': 'Situação',
            'condicaoEleitoral': 'Condição Eleitoral',
            'descricaoStatus': 'Descrição Status'
        })

        st.dataframe(deputado_historico, hide_index=True)

with tab2:
    col1_tab2, col2_tab2, col3_tab2 = st.columns([2, 2, 2])

    with col1_tab2:
        total_despesas = deputado_despesas['valorDocumento'].sum()
        despesas_formatado = f"R$ {total_despesas:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        st.metric("Total de despesas", despesas_formatado)

    with col2_tab2:
        # Verifica se 'deputado_despesas' não está vazio antes de procurar pela despesa mais alta
        if not deputado_despesas.empty:
            despesa_alta = deputado_despesas["valorDocumento"].max()
            despesa_alta_formatada = f"R$ {despesa_alta:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            st.metric("Despesa mais alta", despesa_alta_formatada)
        else:
            st.metric("Despesa mais alta", "Não disponível")

    with st.container():
        if not deputado_despesas.empty:
            fig1 = px.bar(
                deputado_despesas,
                x="mes",
                y="valorDocumento",
                title="Despesas por tipo e mês",
                labels={"mes": "Mês", "valorDocumento": "Valor (R$)", "tipoDespesa": "Tipo de despesa"},
                color="tipoDespesa"
            )
            fig1.update_layout(barmode='stack')
            st.plotly_chart(fig1)
        else:
            st.write("Sem dados de despesas disponíveis para criar o gráfico")

    with st.container():
        col1_tab2_despesas, col2_tab2_fornecedores = st.columns([2, 2])
        # Selecionando as colunas desejadas
        df_despesas = deputado_despesas[[
            "codLote", "codDocumento", "tipoDocumento",
            "codTipoDocumento", "numDocumento", "dataDocumento",
            "valorDocumento", "valorLiquido", "tipoDespesa",
            "urlDocumento", "nomeFornecedor"
        ]]

        with col1_tab2_despesas:
            # Multiselect para selecionar o tipo de despesa
            tipo_despesa = sorted(df_despesas["tipoDespesa"].unique().tolist())
            select_tipo_despesa = st.multiselect("Tipo - Despesa", tipo_despesa)

            # Filtrando as despesas pelas despesas selecionadas
            if select_tipo_despesa:
                df_despesas = df_despesas[df_despesas["tipoDespesa"].isin(select_tipo_despesa)]
            else:
                df_despesas = df_despesas[df_despesas["tipoDespesa"].isin(tipo_despesa)]

        with col2_tab2_fornecedores:
            # Multiselect para selecionar o(s) fornecedor(es)
            fornecedores = sorted(df_despesas["nomeFornecedor"].unique().tolist())
            select_fornecedor = st.multiselect("Fornecedor", fornecedores)

            # Filtrando as despesas pelos fornecedores selecionados
            if select_fornecedor:
                df_despesas = df_despesas[df_despesas["nomeFornecedor"].isin(select_fornecedor)]
            else:
                df_despesas = df_despesas[df_despesas["nomeFornecedor"].isin(fornecedores)]

        # Formatando o df_despesas["valorDocumento"] para R$
        df_despesas["valorDocumento"] = df_despesas["valorDocumento"].apply(
            lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )

        st.dataframe(
            df_despesas,
            column_config={
                "codLote": st.column_config.TextColumn(
                    "Código - Lote"
                ),
                "codDocumento": st.column_config.TextColumn(
                    "Código - Documento"
                ),
                "tipoDocumento": st.column_config.TextColumn(
                    "Tipo - Documento"
                ),
                "codTipoDocumento": st.column_config.TextColumn(
                    "Código - Tipo Documento"
                ),
                "numDocumento": st.column_config.TextColumn(
                    "N° - Documento"
                ),
                "dataDocumento": st.column_config.DateColumn(
                    "Data - Despesa",
                    format="DD/MM/YYYY"
                ),
                "valorDocumento": st.column_config.TextColumn(
                    "Valor - Documento"
                ),
                "valorLiquido": st.column_config.TextColumn(
                    "Valor - Líquido"
                ),
                "tipoDespesa": st.column_config.TextColumn(
                    "Tipo - Despesa"
                ),
                "nomeFornecedor": st.column_config.TextColumn(
                    "Nome - Fornecedor"
                ),
                "urlDocumento": st.column_config.LinkColumn(
                    "Link - Documento",
                    display_text="Abrir Documento"
                )
            },
            hide_index=True,
            width=1500
        )

with tab3:
    col1_tab3, col2_tab3 = st.columns([1, 2])
    with col1_tab3:
        # Total de fornecedores
        if not deputado_despesas.empty:
            total_fornecedores = deputado_despesas['nomeFornecedor'].nunique()
            st.metric("Total de Fornecedores", total_fornecedores)
        else:
            st.metric("Total de Fornecedores", "Não disponível")

        # Fornecedor com maior despesa
        if not deputado_despesas.empty and not deputado_despesas['nomeFornecedor'].isna().all():
            fornecedor_maior_despesa = deputado_despesas.groupby('nomeFornecedor')['valorDocumento'].sum().idxmax()
            st.metric("Fornecedor com maior despesa", fornecedor_maior_despesa)
        else:
            st.metric("Fornecedor com maior despesa", "Não disponível")

    with col2_tab3:
        if not deputado_despesas.empty:
            # Total de despesas por fornecedor
            total_despesas_fornecedor = deputado_despesas.groupby('nomeFornecedor')['valorDocumento'].sum().reset_index()
            total_despesas_fornecedor = total_despesas_fornecedor.sort_values(by='valorDocumento', ascending=False)
            total_despesas_fornecedor['valorDocumento'] = total_despesas_fornecedor['valorDocumento'].apply(
                lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )

            st.dataframe(
                total_despesas_fornecedor,
                column_config={
                    "nomeFornecedor": st.column_config.TextColumn(
                        "Nome - Fornecedor"
                    ),
                    "valorDocumento": st.column_config.TextColumn(
                        "Valor - Documento"
                    )
                },
                hide_index=True,
                width=1300
            )
        else:
            st.write("Sem dados de fornecedores disponíveis")
