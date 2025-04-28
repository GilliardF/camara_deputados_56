
# Análise de Despesas com PCA

Análise de Despesas com PCA é um aplicativo desenvolvido em Python com Streamlit, projetado para facilitar a identificação de padrões e anomalias em dados financeiros de despesas públicas. A análise é feita por meio de Componentes Principais (PCA), proporcionando uma visualização de dados em uma interface interativa.

## Funcionalidades

Análise Exploratória: Visualize histogramas individuais das variáveis e matrizes de correlação para compreender melhor a distribuição e inter-relações nos dados.

- Análise Paralela: Utilize simulações paralelas para determinar quantos componentes principais devem ser utilizados na análise PCA.

- Detecção de Anomalias: Identifique registros fora do esperado, categorizando-os como anômalos ou normais, baseado nos componentes principais.

- Visualização de Controle: Empregue cartas de monitoramento e representações gráficas para validar os resultados obtidos.

- Exportação de Dados: Salve os dados analisados em arquivos CSV para processamento ou avaliação futuros.
## Requisitos

- Python 3.12 ou superior

  https://www.python.org/

- Poetry para gerenciamento de dependências e ambiente

  https://python-poetry.org/
  
## Instalação

Para instalar e configurar o projeto localmente usando o Poetry, siga estas etapas:

Clone o repositório para sua máquina:
    
    git clone https://github.com/GilliardF/camara_deputados_56.git

    cd camara_deputados_56

Ative o ambiente virtual do Poetry:

    poetry shell

Se necessário, criar um novo ambviente virtual e adicionar depedências pelo poetry (configuração manual):

    poetry init
    poetry add


## Uso

Depois de configurado, você pode executar o aplicativo com o seguinte comando:

Certifique-se de que o arquivo de dados data.db está presente na pasta data.

Execute o aplicativo Streamlit:

    streamlit run 1_🏠_Home.py
