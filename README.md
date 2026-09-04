# Laboratórios & Trabalho Final de Ciência de Dados

Este repositório reúne as atividades práticas desenvolvidas ao longo da disciplina eletiva de **Ciência de Dados (Data Science)**, contendo **6 laboratórios práticos** e o **trabalho final** da disciplina.

---

## 👥 Integrantes

* Cristopher de Wallau(https://github.com/CristopherWallau)
* Lucas Vieira Bagolin(https://github.com/LucasBagolin)
* Arthur Andrade (https://github.com/ArthurAndradee)
* Vitor Feijó (https://github.com/vitorsfeijo

---

## 🎯 Trabalho Final: Análise Preditiva e NLP com o Dataset da Olist

O projeto final consistiu em uma análise aprofundada da base de dados pública de e-commerce brasileiro da **Olist** (*Brazilian E-Commerce Public Dataset by Olist*), focando na experiência do consumidor e na satisfação pós-compra. 

O trabalho foi dividido em duas abordagens preditivas complementares:

### 1. Previsão de Avaliação por Linguagem Natural (NLP)
* **Objetivo:** Prever a nota final (*review score* de 1 a 5 estrelas) a partir dos comentários e textos deixados pelos clientes nas avaliações.
* **Técnicas:**
  * Pré-processamento de texto em português (remoção de stopwords, pontuação, normalização/tokenização).
  * Vetorização textual (TF-IDF / Bag of Words / Embeddings).
  * Modelagem de classificação/regressão para mapear o sentimento do texto à nota correspondente.

### 2. Previsão de Avaliação com Base em Métricas Logísticas
* **Objetivo:** Identificar o impacto de variáveis operacionais na nota de avaliação da entrega, prevendo insatisfações antes mesmo do feedback textual.
* **Variáveis exploradas:**
  * Prazo de entrega estimado vs. data de entrega real.
  * Ocorrência e gravidade de atrasos logísticos.
  * Valor do frete em relação ao valor do produto.
  * Distância logística e tempo de postagem/despacho.
* **Técnicas:** Engenharia de atributos (criação de *features* temporais e de atraso), tratamento de nulos, normalização e modelos preditivos supervisionados.

---

## 🧪 Laboratórios (Labs 1 a 6)

O repositório também inclui as soluções desenvolvidas para os 6 laboratórios práticos da disciplina, abordando fundamentos e etapas essenciais do ciclo de vida de projetos de Ciência de Dados:

* **Lab 01:** Coleta, carregamento e visualização de dados com Pandas.
* **Lab 02:** Análise exploratória de dados (EDA) e feature engineering.
* **Lab 03:** Pré-processamento, limpeza de dados e compreensão de regressão e overfitting/underfitting.
* **Lab 04:** Modelos supervisionados de classificação e métricas de desempenho.
* **Lab 05:** Modelos de regressão e validação cruzada para avaliar o dataset Ames Housing para prever preços de casas.
* **Lab 06:** Agrupamento não-supervisionado (Clustering)/Técnicas avançadas.

---

## 📁 Estrutura de Pastas

```text
├── labs/
│   ├── lab_01.ipynb
│   ├── lab_02.ipynb
│   ├── lab_03.ipynb
│   ├── lab_04.ipynb
│   ├── lab_05.ipynb
│   └── lab_06.ipynb
├── final_project/
│   ├── data/                 # Conjuntos de dados ou instruções de download
│   ├── final_project.ipynb   # Notebook principal do Trabalho Final
│   └── reports/              # Gráficos e resultados exportados
├── requirements.txt          # Dependências do projeto
└── README.md
