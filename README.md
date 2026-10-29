# Rede Neural para Classificação Binária em Dados Tabulares

Código-fonte do pipeline experimental do artigo **"Desenvolvimento da Inteligência Artificial: estruturação do banco de dados, treinamento, codificação da IA, testes e validação"**, desenvolvido na Universidade Evangélica de Goiás (UniEVANGÉLICA).

---

## Sobre o projeto

Este repositório investiga como técnicas de arquitetura, regularização e otimização afetam o desempenho e a estabilidade de uma rede neural aplicada a dados tabulares. O estudo parte de um Multilayer Perceptron (MLP) básico e incorpora, de forma progressiva, mecanismos de atenção, conexões residuais, normalização por camadas e função de perda focal.

---

## Fases experimentais

| Fase | Descrição | Execuções |
|:---:|---|:---:|
| **1** | MLP basal (ReLU + Binary Cross Entropy) | 30 |
| **2** | Gated Feature Attention + conexões residuais | 30 |
| **3** | Base enriquecida (24 variáveis) + ruído gaussiano no treino | 30 |
| **4** | Binary Focal Loss + LayerNorm + SiLU (e teste de repetibilidade) | 30 |
| **5** | Configuração final com precisão mista automática (AMP) | 35 |

---

## Base de dados

- **1.000 registros**, 20 variáveis numéricas originais + 4 derivadas por engenharia de atributos (24 preditores) e uma coluna de saída binária.
- **Divisão estratificada**: 70% treino, 15% validação, 15% teste.
- **`RobustScaler`** ajustado somente no conjunto de treino, para evitar vazamento de dados (*data leakage*).
- **Ruído gaussiano** (\(\sigma = 0,01\)) aplicado exclusivamente no conjunto de treino para aumentar a robustez.

---

## Treinamento

- **Otimizador**: AdamW (weight decay = 1e-4)
- **Taxa de aprendizado**: Cosine Annealing Warm Restarts
- **Gradient clipping**: 1,0
- **Early Stopping**: paciência de 15 épocas sobre a perda de validação
- **Precisão mista automática (AMP)**: `torch.amp.GradScaler`
- **Função de Perda**: Binary Focal Loss com \(\gamma = 2,0\)

---

## Resultados principais (Fase 5, 35 execuções)

| Métrica | Média ± desvio padrão |
|---|:---:|
| **Acurácia** | 95,22% ± 0,98% |
| **F1-Score** | 95,21% ± 0,99% |
| **AUC-ROC** | 0,9856 ± 0,0044 |
| **Perda de validação** | 0,0272 ± 0,0058 |

> **Conclusão:** O maior benefício de aumentar a complexidade da rede foi a **redução expressiva da variabilidade entre execuções** (estabilidade estatística), e não apenas o aumento pontual das médias de acerto.

---

## Estrutura do Repositório

```text
├── data/
│   ├── raw/
│   │   └── dataset.csv              # Base tabular bruta com features e target
│   └── processed/
│       ├── scaler.joblib            # Scaler ajustado para inferência
│       └── resultados_*.csv         # Históricos das baterias experimentais (Fases 1 a 5)
├── models/
│   └── best_model.pt                # Checkpoint do melhor modelo treinado
├── src/
│   ├── __init__.py
│   ├── config.py                    # Hiperparâmetros, seeds e caminhos do projeto
│   ├── dataset.py                   # Pipeline de dados, RobustScaler e engenharia de features
│   ├── model.py                     # Arquitetura ClassifierMLP (Gated Attention + Residual Blocks)
│   ├── train.py                     # Treinamento individual com Early Stopping
│   ├── evaluate.py                  # Avaliação no split de teste (Acurácia, F1, ROC-AUC, CM)
│   ├── run_30_experiments.py        # Bateria de 30 experimentos
│   ├── run_35_experiments.py        # Bateria da Fase 5 (35 ciclos com AMP)
│   └── main_api.py                  # API REST com FastAPI para predição em tempo real
├── requirements.txt                 # Dependências do projeto
├── LICENSE                          # Licença MIT
└── README.md
```

---

## Como executar

### 1. Clonar o repositório e preparar o ambiente
```bash
git clone https://github.com/IsaqueTF/rede-neural-dados-tabulares.git
cd rede-neural-dados-tabulares

python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Executar o Treinamento
```bash
python -m src.train
```

### 3. Avaliar no Conjunto de Teste
```bash
python -m src.evaluate
```

### 4. Reproduzir a Bateria Experimental (Fase 5 - 35 Ciclos)
```bash
python -m src.run_35_experiments
```

### 5. Executar a API de Inferência
```bash
uvicorn src.main_api:app --host 0.0.0.0 --port 8000 --reload
```
Acesse a documentação Swagger em: `http://localhost:8000/docs`

---

## Tecnologias

- **Linguagem**: Python
- **Deep Learning**: PyTorch
- **Machine Learning & Pré-processamento**: Scikit-Learn
- **Manipulação de Dados**: Pandas, NumPy
- **Deploy / API**: FastAPI, Uvicorn, Pydantic

---

## Limitações

Base com 1.000 instâncias na formulação experimental, o que restringe a generalização direta para bases massivas. Trabalhos futuros incluem testes com dados reais altamente desbalanceados e estudo comparativo aprofundado com modelos baseados em árvores de decisão como XGBoost e LightGBM.

---

## Autores

- **Isaque Teles Ferreira**
- **Igor Dalarmelino Borges**

*Universidade Evangélica de Goiás (UniEVANGÉLICA)*

---

## Licença

Distribuído sob a licença MIT. Veja o arquivo [`LICENSE`](LICENSE) para mais detalhes.
