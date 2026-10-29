# Rede Neural para Dados Tabulares

Pipeline completo de aprendizado profundo (Deep Learning) voltado para classificação em dados tabulares complexos, integrando engenharia de atributos, mecanismo de atenção (*Gated Feature Attention*), blocos residuais (*ResNet-style*) e uma API de inferência em tempo real com FastAPI.

---

## 🎯 Objetivos do Projeto

1. **Superação de MLPs Tradicionais em Dados Tabulares**: Implementar técnicas modernas de deep learning adequadas à natureza de dados tabulares (normalização por camada, ativação suave SiLU, atenção de features e skip-connections).
2. **Robustez a Ruído e Outliers**: Emprego de `RobustScaler` e engenharia de características não-lineares, além de função de perda `Binary Focal Loss` focada em instâncias difíceis.
3. **Reprodutibilidade e Validação Estatística**: Baterias de testes com 30 a 35 ciclos independentes com sementes randômicas para garantir que as métricas de acurácia, F1-Score e AUC-ROC sejam consistentes.
4. **Deploy Pronto para Produção**: Exposição do modelo e dos artefatos de pré-processamento via API REST assíncrona de baixa latência (`FastAPI`).

---

## 🏗️ Arquitetura do Modelo

O modelo implementado em `src/model.py` (`ClassifierMLP`) possui os seguintes componentes:

```
[ Entrada (Features Tabulares) ]
               │
               ▼
   [ Gated Feature Attention ]   <-- Ponderação adaptativa de atributos
               │
               ▼
   [ Camada de Projeção Linear ]
   [ LayerNorm + SiLU + Dropout ]
               │
               ▼
   [ Stable Residual Block 1 ]   <-- Conexão residual balanceada (x + 0.5 * F(x))
               │
               ▼
   [ Stable Residual Block 2 ]
               │
               ▼
   [ Cabeça de Classificação ]
               │
               ▼
       [ Logits / Sigmoid ]
```

### Principais Diferenciais:
- **Gated Feature Attention**: Aplica uma máscara suave de gating sobre o vetor de entrada para destacar variáveis informativas e atenuar ruídos.
- **StableResidualBlock**: Combina Linear, LayerNorm, SiLU e Dropout, com soma ponderada residual para estabilizar o gradiente.
- **Otimização**: Otimizador `AdamW` com decaimento de taxa de aprendizado cíclico (`CosineAnnealingWarmRestarts`) e precisão mista automática (AMP via `torch.amp`).

---

## 📊 Métricas e Resultados

Nas baterias de testes empíricos de 35 execuções independentes (`resultados_35_experimentos_alteracao5.csv`):
- **Acurácia Média**: ~95.5% a 97.0%
- **F1-Score**: ~0.95 a 0.97
- **AUC-ROC**: > 0.985 (alcançando até 0.992)
- **Loss de Validação**: ~0.023 a 0.035

---

## 📁 Estrutura do Repositório

```text
├── data/
│   ├── raw/
│   │   └── dataset.csv              # Base tabular bruta com features e target
│   └── processed/
│       ├── scaler.joblib            # Scaler ajustado para inferência
│       └── resultados_*.csv         # Históricos das baterias experimentais
├── models/
│   └── best_model.pt                # Pesos do melhor modelo treinado
├── src/
│   ├── __init__.py
│   ├── config.py                    # Hiperparâmetros e caminhos do projeto
│   ├── dataset.py                   # Pipeline de dados, split estratificado e engenharia
│   ├── model.py                     # Arquitetura ClassifierMLP e Feature Attention
│   ├── train.py                     # Script de treinamento de ciclo único com Early Stopping
│   ├── train_multiple.py            # Execução de múltiplos treinamentos
│   ├── evaluate.py                  # Avaliação no split de teste (Acurácia, F1, ROC-AUC, CM)
│   ├── run_30_experiments.py        # Bateria de 30 experimentos
│   ├── run_35_experiments.py        # Bateria avançada de 35 experimentos com AMP
│   └── main_api.py                  # API REST com FastAPI para predição
├── requirements.txt                 # Dependências do projeto
└── README.md
```

---

## 🚀 Como Executar

### 1. Instalação do Ambiente
```bash
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Treinamento do Modelo
```bash
python -m src.train
```

### 3. Avaliação no Conjunto de Teste
```bash
python -m src.evaluate
```

### 4. Bateria de Experimentos
```bash
python -m src.run_35_experiments
```

### 5. Execução da API de Inferência
```bash
uvicorn src.main_api:app --host 0.0.0.0 --port 8000 --reload
```
Acesse a documentação interativa Swagger em: `http://localhost:8000/docs`

#### Exemplo de Requisição (POST `/predict`):
```json
{
  "features": [0.655, -0.605, -2.714, 0.825, -0.796, -1.237, 3.152, -7.691, 0.648, 1.716, -1.318, 1.780, 1.129, -2.174, -0.131]
}
```
Resposta:
```json
{
  "probability": 0.9412,
  "predicted_class": 1
}
```
