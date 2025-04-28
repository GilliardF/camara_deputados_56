
# Análise com PCA referente as despesas dos Deputados em 2022

O aplicativo "camara_deputados_56" é uma ferramenta intuitiva e desenvolvida em Python com o uso do Streamlit. Ele facilita a navegação através dos dados financeiros relacionados às despesas públicas dos deputados em 2022. Utilizando a técnica de Análise de Componentes Principais (PCA), o aplicativo destaca padrões e anomalias nos dados. Com uma interface interativa e fácil de usar, oferece uma maneira acessível de compreender como esses recursos foram utilizados ao longo do ano, proporcionando insights para entender possíveis anomalias.

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

Se necessário, criar um novo ambiente virtual e adicionar depedências pelo poetry (configuração manual):

    poetry init
    poetry add


## Uso

Depois de configurado, você pode executar o aplicativo com o seguinte comando:

Certifique-se de que o arquivo de dados data.db está presente na pasta data.

Execute o aplicativo Streamlit:

    streamlit run 1_🏠_Home.py
