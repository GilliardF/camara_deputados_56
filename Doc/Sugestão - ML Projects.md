# **1. Análise Distributiva de Despesas (Região, Partido, Deputado, Fornecedor…)**

Aqui, vamos focar na distribuição dos gastos por diferentes aspectos.

### **1.1. Distribuição dos Gastos por Região**

- Somar os valores gastos por cada estado (**siglaUf**) e comparar.

- Calcular o **gasto médio por deputado** em cada estado.

- Criar um **top 5 de estados com maiores/menores despesas**.

- Visualizar a distribuição de **quais tipos de despesas são mais comuns em cada região** (exemplo: estados do Norte gastam mais com passagens aéreas?).

**Objetivo para ML:**

- Pode ser útil para detectar se certos estados são mais propensos a gastos altos.

### **1.2. Distribuição dos Gastos por Partido**

- Calcular o **gasto médio por deputado** dentro de cada partido (**siglaPartido**).

- Comparar **quais partidos gastam mais em determinados tipos de despesas** (exemplo: um partido gasta mais com publicidade do que outros?).

- Criar um ranking dos **partidos mais gastadores**.

**Objetivo para ML:**

- Pode servir para identificar padrões de gastos por partidos, sugerindo traços administrativos específicos.

### **1.3. Distribuição dos Gastos por Deputado**

- Identificar deputados com **maiores/menores valores gastos**.

- Determinar **gastos médios mensais/anuais por deputado**.

- Detectar **deputados com gastos anômalos** (exemplo: um deputado que sempre gasta muito acima da média).

**Objetivo para ML:**

- Está relacionado à detecção de outliers (gastos suspeitos).

### **1.4. Distribuição dos Gastos por Tipo de Despesa**

- Identificar os **tipos de despesa mais usados**.

- Descobrir **quais tipos de despesa têm valores médios mais altos**.

- Verificar se há um aumento ou queda em determinados **tipos de gasto ao longo dos meses/anos**.

**Objetivo para ML:**

- Pode ajudar a prever tendências de gastos futuros para diferentes categorias.

### **1.5. Distribuição dos Gastos por Fornecedor**

- Descobrir **quais fornecedores receberam mais dinheiro**.

- Comparar se alguns fornecedores concentram a maior parte das despesas (sinal de possível irregularidade).

- Identificar fornecedores com altos valores de **despesas atreladas a um único deputado** (favoritismo).

- Verificar se os mesmos fornecedores são usados por diferentes partidos ou se há preferência por determinados partidos.

**Objetivo para ML:**

- Modelos de detecção de anomalias podem ser treinados para encontrar fornecedores com características incomuns.

## **2. Análise Temporal das Despesas**

Aqui, analisamos as flutuações dos gastos ao longo do tempo.

### **2.1. Evolução dos Gastos ao Longo dos Meses e Anos**

- Criar séries temporais de **valor total gasto por mês e por ano**.

- Comparar se há **crescimento ou diminuição dos gastos ao longo do tempo**.

- Identificar **épocas do ano com aumento de gastos** (exemplo: há picos em anos eleitorais?).

**Objetivo para ML:**

- Algoritmos de **séries temporais** podem prever gastos futuros e identificar sazonalidade nos gastos parlamentares.

### **2.2. Identificação de Sazonalidade nos Gastos**

- Existe um **padrão de aumento de gastos em determinados meses do ano**?

- Deputados recém-eleitos gastam mais no começo do mandato?

- Há diferença nos gastos **antes ou depois de recessos parlamentares**?

**Objetivo para ML:**

- Detectar padrões temporais e prever se em determinados períodos os gastos são suspeitos ou esperados.

## **3. Detecção de Anomalias e Possíveis Fraudes**

Aqui focamos nos comportamentos fora do padrão.

### **3.1. Gastos Extremamente Altos para um Único Deputado**

- Encontrar parlamentares que gastam consistentemente **acima da média**.

- Avaliar se o padrão de gastos deles **desvia muito do normal em relação aos colegas do mesmo estado/partido**.

**Objetivo para ML:**

- Pode ser útil para modelos de **detecção de outliers**, como **Isolation Forest ou DBSCAN**.

### **3.2. Gastos Incompatíveis com Períodos de Viagem**

- Deputados justificam altos gastos com viagens, mas **não foram registrados fora de Brasília nesses períodos**?

- Gastos com deslocamento correspondem a períodos de sessões presenciais?

**Objetivo para ML:**

- Pode ser cruzado com bases externas para identificar fraudes potenciais.

### **3.3. Deputados com Gastos Elevados em Fornecedores "exclusivos"**

- Empresas que **aparecem apenas para um único deputado** e recebem grandes valores.

- Deputados que concentram altos valores em **CNPJs de empresas recém-abertas**.

**Objetivo para ML:**

- Usar modelos supervisionados para encontrar padrões de **gastos suspeitos**.

### **3.4. Comparação de Gastos Declarados vs. Valores Habitualmente Praticados no Mercado**

- Um deputado gasta R$ 500 em um jantar simples enquanto restaurantes similares cobram R$ 100?

- Detectar valores **significativamente inflacionados**.

**Objetivo para ML:**

- Modelos de regressão e análise estatística podem ajudar a determinar padrões que sugerem superfaturamento.

# **Conclusão: Como isso se encaixa em Machine Learning?**

Essa análise exploratória ajudará a fazer um **pré-processamento** e levantar **hipóteses** para o uso de Machine Learning. Aqui estão alguns exemplos de modelos úteis para o seu caso:

✅ **Modelos de Séries Temporais** (ARIMA, Prophet) → Projeção de gastos futuros

✅ **Modelos de Detecção de Outliers** (Isolation Forest, DBSCAN) → Identificação de gastos anômalos

✅ **Modelos Supervisionados (Classificação)** → Criar rótulos de "gasto suspeito" e treinar um modelo para prever fraudes

✅ **Modelos Não Supervisionados (Agrupamento - K-Means, GMM)** → Encontrar padrões em grupos de deputados com perfis de gasto similares