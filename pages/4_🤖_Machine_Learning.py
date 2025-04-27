import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f, chi2
from data.despesas_ML import ML_despesas
import os
from dateutil.parser import parse

# ------------- INÍCIO APP STREAMLIT -------------

# Configuração da página
st.set_page_config(
    page_title="Machine Learning",
    layout="wide",
    page_icon="🤖"
)

# -------- Funções auxiliares --------

@st.cache_data
def load_data(db_path):
    df = ML_despesas(db_path)

    # Certificar-se de que 'dataDocumento' é interpretado como data
    if 'dataDocumento' in df.columns:
        try:
            df['dataDocumento'] = pd.to_datetime(df['dataDocumento'], errors='coerce')
            df = df.dropna(subset=['dataDocumento'])  # Remover datas inválidas
        except Exception as e:
            st.warning(f"Erro ao converter as datas: {e}")
            df['dataDocumento'] = pd.NaT

    return df

def calc_pca_scores_and_monitoring(df, numeric_cols, pca_obj, scaler_obj):
    X = df[numeric_cols].values
    X_scaled = scaler_obj.transform(X)
    Xpca = pca_obj.transform(X_scaled)
    var_expl = pca_obj.explained_variance_
    T2 = np.sum((Xpca**2) / var_expl, axis=1)
    Xpca_inv = pca_obj.inverse_transform(Xpca)
    SPE = np.sum((X_scaled - Xpca_inv) ** 2, axis=1)
    return T2, SPE

def hotelling_T2_limit(n_samples, n_components, alpha=0.95):
    return n_components * (n_samples - 1) / (n_samples - n_components) * f.ppf(alpha, n_components, n_samples - n_components)

def SPE_limit(X_train_scaled, pca, alpha=0.95):
    recon_train = pca.inverse_transform(pca.transform(X_train_scaled))
    SPEs = np.sum((X_train_scaled - recon_train) ** 2, axis=1)
    limit = np.percentile(SPEs, 100*alpha)
    return limit

def parallel_analysis(X, n_iter=50, random_state=42, n_jobs=4):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    real_eig = pca.explained_variance_

    # Função interna para simulação aleatória
    def random_simulation():
        np.random.seed(random_state)
        X_rand = np.copy(X)
        for col in range(X.shape[1]):
            np.random.shuffle(X_rand[:, col])
        X_rand_scaled = scaler.transform(X_rand)
        pca_rand = PCA()
        pca_rand.fit(X_rand_scaled)
        return pca_rand.explained_variance_

    # Processamento paralelo
    rand_eigs = Parallel(n_jobs=n_jobs)(delayed(random_simulation)() for _ in range(n_iter))
    mean_rand_eig = np.mean(rand_eigs, axis=0)
    return real_eig, mean_rand_eig

def pca_anomaly_detection(df, numeric_cols, n_components=1, contamination=0.05, pca_train_df=None):
    X = df[numeric_cols].values
    if pca_train_df is not None:
        X_train = pca_train_df[numeric_cols].values
    else:
        X_train = X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled = scaler.transform(X)
    X_train_scaled = scaler.transform(X_train)
    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)
    X_pca = pca.transform(X_scaled)
    anomaly_scores = np.abs(X_pca[:, 0])
    threshold = np.percentile(np.abs(pca.transform(X_train_scaled)[:, 0]), 100 * (1 - contamination))
    anomalies = anomaly_scores > threshold
    result = df.copy()
    result['AnomalyScore'] = anomaly_scores
    result['IsAnomaly'] = anomalies.astype(int)
    return result, threshold, pca, scaler, X_train_scaled

def simula_dados_normais(df, numeric_cols, n_samples=300, random_state=1001):
    X = df[numeric_cols].values
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    np.random.seed(random_state)
    simulados = np.random.multivariate_normal(mean, cov, size=n_samples)
    return pd.DataFrame(simulados, columns=numeric_cols)

def add_hotelling_ellipse(fig, mean, cov, level=0.95, name="Elipse Hotelling T²", color="limegreen"):
    q = chi2.ppf(level, df=2)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals * q)
    t_ = np.linspace(0, 2*np.pi, 250)
    ell = np.array([width/2 * np.cos(t_), height/2 * np.sin(t_)])
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    ell_rot = R @ ell
    xs = ell_rot[0, :] + mean[0]
    ys = ell_rot[1, :] + mean[1]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines", name=name, line=dict(color=color, dash="dash"),
        showlegend=True
    ))
    return fig

def grafico_contribuicao(pca, scaler, df, numeric_cols):
    X_scaled = scaler.transform(df[numeric_cols].values)
    componentes = pca.components_
    contribs = componentes[0]**2
    contribs = contribs / contribs.sum()
    fig = px.bar(x=numeric_cols, y=contribs,
                 labels={'x': 'Variável', 'y': 'Contribuição para 1º Componente'},
                 title='Contribuição das variáveis no 1º componente principal')
    return fig

st.title("Detectar Anomalias de despesas com PCA")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Análise Exploratória", "📈 Análise Paralela", "📊 Detecção de Anomalias", "🎯 Validando o Modelo"])

# Localização do banco de dados
db_path = os.path.abspath("data/data.db")

try:
    with st.spinner('Carregando dados do banco de dados...'):
        df = load_data(db_path)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Análise Exploratória
    with tab1:
        col1, col2 = st.columns([2, 2])
        with col1:
            for col in numeric_cols:
                st.subheader(f"Histograma de {col}")
                fig_hist = px.histogram(df, x=col, nbins=30, color_discrete_sequence=['dodgerblue'])
                fig_hist.update_layout(bargap=0.05, xaxis_title=col, yaxis_title="Frequência")
                st.plotly_chart(fig_hist, use_container_width=True)
                
        with col2:
            st.write(df.describe())
            st.subheader("Correlação entre variáveis numéricas")
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr.values,
                    text_auto=True,
                    x=numeric_cols,
                    y=numeric_cols,
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1,
                    aspect='auto'
                )
                fig_corr.update_layout(width=400, height=400, coloraxis_colorbar=dict(title="Correlação"))
                st.plotly_chart(fig_corr, use_container_width=True)

    # Análise Paralela
    with tab2:
        col1_tab2, col2_tab2 = st.columns(2)
        st.header("Análise Paralela")
        st.subheader("Parallel Analysis (Quantos componentes principais reter?)")

        with st.spinner('Executando análise paralela...'):
            X_features = df[numeric_cols].dropna().values
            real_eig, rand_eig = parallel_analysis(X_features, n_iter=100, n_jobs=4)

        x = list(range(1, len(real_eig) + 1))

        fig_pa = go.Figure()
        fig_pa.add_trace(go.Scatter(x=x, y=real_eig, mode='lines+markers', name="Autovalores reais"))
        fig_pa.add_trace(go.Scatter(x=x, y=rand_eig, mode='lines+markers', name="Média autovalores aleatórios"))
        fig_pa.update_layout(
            title='Número ótimo de componentes PCA',
            xaxis_title='Componente Principal', yaxis_title='Autovalor'
        )
        st.plotly_chart(fig_pa, use_container_width=True)
        n_sug = int(np.sum(real_eig > rand_eig))
        with col1_tab2:
            st.info(f"Número sugerido de componentes principais para PCA: **{n_sug}**")
        
        with col2_tab2:
            n_components = st.slider("Quantos componentes principais usar no PCA?",
                                     min_value=1, max_value=len(real_eig), value=n_sug)

    # Detecção de Anomalias
    with tab3:
        col1_tab3, col2_tab3 = st.columns([2, 8])

        with col1_tab3:
            st.write("Colunas usadas para análise:", numeric_cols)
            min_max = {}
            for col in numeric_cols:
                min_max[col] = (
                    float(df[col].quantile(0.05)),
                    float(df[col].quantile(0.95))
                )
        with col2_tab3:
            sliders = {}
            for col in numeric_cols:
                minval, maxval = float(df[col].min()), float(df[col].max())
                sliders[col] = st.slider(
                    f"Intervalo de valores considerados normais para '{col}' (para treino do PCA):",
                    min_value=minval, max_value=maxval, value=min_max[col], key=f"slider_{col}"
                )
        
        st.divider()
        
        from functools import reduce
        conds = [(df[col] >= sliders[col][0]) & (df[col] <= sliders[col][1]) for col in numeric_cols]
        filtro = reduce(lambda a, b: a & b, conds)
        dados_treino = df[filtro]

        contamination = 0.05

        # Detecção de Anomalias
        resultados, threshold, pca_fit, scaler_fit, X_train_scaled = pca_anomaly_detection(
            df, numeric_cols=numeric_cols, n_components=n_components,
            contamination=contamination, pca_train_df=dados_treino
        )

        col1a_tab3, col2a_tab3 = st.columns(2)
        with col1a_tab3:
            # Dados Reais
            st.markdown(f'Quantidade de registros "normais": {len(resultados[resultados["IsAnomaly"] == 0])}')
            st.markdown(f'Quantidade de registros "anômalos": {len(resultados[resultados["IsAnomaly"] == 1])}')

            # Filtrar apenas as anomalias
            tipo_anomalies = resultados[resultados['IsAnomaly'] == 1]

            # Contar número de anomalias por tipo de despesa
            tipo_despesa_count = tipo_anomalies['tipoDespesa'].value_counts().sort_values(ascending=True)

            # Criar gráfico de barras horizontal
            fig_bar = px.bar(
                y=tipo_despesa_count.index,
                x=tipo_despesa_count.values,
                labels={'y': 'Tipo de Despesa', 'x': 'Total de Anomalias'},
                title='Total de Anomalias por Tipo de Despesa',
                orientation='h'
            )

            # Adicionar rótulos de dados
            fig_bar.update_traces(text=tipo_despesa_count.values, textposition='auto')
            
            # Atualizar layout deixando o gráfico mais alto em tamanho
            fig_bar.update_layout(
                height=950,
                showlegend=False,
                xaxis_title="Total de Anomalias",
                yaxis_title="Tipo de Despesa"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        with col2a_tab3:
            st.success(f"Detecção concluída. Limiar (Threshold): {threshold:.2f}")

            # Gráfico de Contribuição
            fig_cont = grafico_contribuicao(pca_fit, scaler_fit, df, numeric_cols)
            st.plotly_chart(fig_cont, use_container_width=True)

            # Contagem de anomalias por UF
            tipo_uf_count = tipo_anomalies['siglaUf'].value_counts().sort_values(ascending=False)

            # Criar gráfico de barras vertical
            fig_bar_uf = px.bar(
                x=tipo_uf_count.index,     # UF no eixo x
                y=tipo_uf_count.values,    # Total de anomalias no eixo y
                labels={'x': 'UF', 'y': 'Total de Anomalias'},
                title='Total de Anomalias por UF'
            )
            fig_bar_uf.update_traces(text=tipo_uf_count.values, textposition='auto')
            fig_bar_uf.update_layout(showlegend=False)
            st.plotly_chart(fig_bar_uf, use_container_width=True)

        # Gráfico de linhas com quantidade de anomalias durante cada mês de dataDocumento
        if 'dataDocumento' in df.columns and not df['dataDocumento'].isnull().all():
            try:
                df['mes'] = df['dataDocumento'].dt.to_period('M').dt.to_timestamp()
                anomalies_by_month = (
                    resultados[resultados['IsAnomaly'] == 1]
                    .groupby(df['mes'])  # Importante: Agrupar por 'mes'
                    .size().reset_index(name='count')
                )

                fig_line = px.line(
                    anomalies_by_month,
                    x='mes',
                    y='count',
                    title='Quantidade de Anomalias por Mês',
                    labels={'mes': 'Mês', 'count': 'Quantidade de Anomalias'},
                    markers=True  # Adicionar marcadores para cada ponto no gráfico
                )
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao processar os dados para análise mensal: {e}")
        else:
            st.warning("A coluna 'dataDocumento' não possui dados válidos para extrair informações mensais.")

    with tab4:
        # CARTA T2 E SPE

        # Limite a visualização da carta de Hotelling T² e SPE a 500 registros
        max_visualization_size = 500
        if len(df) > max_visualization_size:
            df_sample = df.sample(n=max_visualization_size, random_state=42)
        else:
            df_sample = df

        # Cálculo de T² e SPE apenas para a amostra
        T2_all, SPE_all = calc_pca_scores_and_monitoring(df_sample, numeric_cols, pca_fit, scaler_fit)
        T2_limit = hotelling_T2_limit(max(len(dados_treino), n_components+1), n_components, alpha=0.95)
        resultados['Hotelling_T2'] = np.nan
        resultados.loc[df_sample.index, 'Hotelling_T2'] = T2_all

        st.subheader("Carta de Monitoramento de Hotelling T²")
        fig_t2 = go.Figure()
        fig_t2.add_trace(go.Scatter(
            y=T2_all, mode='lines+markers', name='T²',
            marker=dict(color=np.where(T2_all > T2_limit, 'crimson', 'royalblue'))
        ))
        fig_t2.add_trace(go.Scatter(
            x=[0, len(T2_all)-1], y=[T2_limit, T2_limit], mode='lines',
            name=f'Limite 95% ({T2_limit:.2f})', line=dict(color='black', dash='dash')
        ))
        fig_t2.update_layout(title="Carta de Hotelling T² (Amostra de até 500 registros)", xaxis_title="Índice",
                             yaxis_title="T²", showlegend=True)
        st.plotly_chart(fig_t2, use_container_width=True)

        st.subheader("Carta do Erro Quadrado de Predição (SPE/Q-residual)")
        spe_threshold = SPE_limit(X_train_scaled, pca_fit, alpha=0.95)
        fig_spe = go.Figure()
        fig_spe.add_trace(go.Scatter(
            y=SPE_all, mode='lines+markers', name='SPE',
            marker=dict(color=np.where(SPE_all > spe_threshold, 'darkorange', 'seagreen'))
        ))
        fig_spe.add_trace(go.Scatter(
            x=[0, len(SPE_all)-1], y=[spe_threshold, spe_threshold], mode='lines',
            name=f'Limite 95% ({spe_threshold:.2f})', line=dict(color='black', dash='dash')
        ))
        fig_spe.update_layout(title="Carta do Erro Quadrado de Predição (SPE/Q-residual)", xaxis_title="Índice",
                              yaxis_title="SPE", showlegend=True)
        st.plotly_chart(fig_spe, use_container_width=True)

        # SCATTER E ELIPSE HOTELLING T2: DADOS REAIS VS SIMULADOS
        st.subheader("Comparação Dados Reais vs Dados Simulados + Elipse de Controle")

        # Define o padrão para colunas 'id_deputado' e 'valorDocumento'
        plot_cols = ['id_deputado', 'valorDocumento']
        if all(col in numeric_cols for col in plot_cols):
            # Simular dados normais
            df_simulado = simula_dados_normais(df, plot_cols, n_samples=300)

            # Limitar a amostragem para os dados reais e simulados na elipse
            df_real_sample = df.sample(n=min(len(df), max_visualization_size), random_state=42)
            df_sim_sample = df_simulado.sample(n=min(len(df_simulado), max_visualization_size), random_state=42)

            scaler_sim = StandardScaler().fit(df_sim_sample[plot_cols])

            X_real = scaler_sim.transform(df_real_sample[plot_cols])
            X_sim = scaler_sim.transform(df_sim_sample[plot_cols])
            mean_sim = np.mean(X_sim, axis=0)
            cov_sim = np.cov(X_sim, rowvar=False)
            fig_ellipse = go.Figure()
            fig_ellipse.add_trace(go.Scatter(
                x=X_sim[:, 0], y=X_sim[:, 1], mode="markers",
                name="Simulados Normais",
                marker=dict(color='deepskyblue', size=7, opacity=0.4, symbol="circle-open"),
                showlegend=True
            ))
            fig_ellipse.add_trace(go.Scatter(
                x=X_real[:, 0], y=X_real[:, 1], mode="markers",
                name="Dados Reais",
                marker=dict(color='orangered', size=8, opacity=0.6, symbol="diamond"),
                showlegend=True
            ))
            fig_ellipse = add_hotelling_ellipse(fig_ellipse, mean_sim, cov_sim, level=0.95, name="Elipse de Controle 95%", color="limegreen")
            fig_ellipse.update_layout(
                title=f'Região Normal (Hotelling T², 95%) – Eixo: {plot_cols[0]} x {plot_cols[1]}',
                xaxis_title=f'{plot_cols[0]} (padronizado)', yaxis_title=f'{plot_cols[1]} (padronizado)'
            )
            st.plotly_chart(fig_ellipse, use_container_width=True)
        else:
            st.warning("As colunas 'id_deputado' e 'valorDocumento' não estão presentes nos dados para visualização na elipse.")

        # Opções de download
        csv_real = resultados.to_csv(index=False).encode()
        st.download_button("Download dos dados reais", csv_real, "dados_reais.csv")

        csv_simulado = df_simulado.to_csv(index=False).encode()
        st.download_button("Download dos dados simulados", csv_simulado, "dados_simulados.csv")
except Exception as e:
    st.error(f"Erro ao carregar os dados do banco de dados: {e}")
