# Rede Neural para Classificação Binária em Dados Tabulares

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Código-fonte oficial do pipeline experimental do artigo científico:  
**"Desenvolvimento da Inteligência Artificial: estruturação do banco de dados, treinamento, codificação da IA, testes e validação"**  
*Universidade Evangélica de Goiás (UniEVANGÉLICA), 2026*

**Autores:**
- **Isaque Teles Ferreira**
- **Igor Dalarmelino Borges**

---

## 🔬 Sobre o Projeto

Este repositório investiga como técnicas avançadas de arquitetura neural, regularização e otimização impactam a estabilidade e o desempenho preditivo de redes neurais profundas aplicadas a **dados tabulares**. 

Partindo de um *Multilayer Perceptron* (MLP) basal, o estudo implementa e avalia de maneira progressiva:
1. **Mecanismo de Atenção por Atributo (*Gated Feature Attention*)**: Ponderação dinâmica da relevância de cada feature na entrada.
2. **Blocos Residuais Estáveis (*ResNet-style*)**: Conexões residuais balanceadas combinadas com `LayerNorm` e função de ativação suave `SiLU`.
3. **Engenharia de Características e Robustez**: Uso de `RobustScaler` ajustado estritamente no conjunto de treino e injeção controlada de ruído gaussiano (σ = 0,01).
4. **Função de Perda Focal (*Binary Focal Loss*)**: Atenuação de exemplos fáceis e estabilização de convergência.
5. **Precisão Mista Automática (AMP)**: Aceleração computacional via `torch.amp.GradScaler` com conservação de estabilidade numérica.

---

## 📊 Tabela 1 do Artigo (Resultados Oficiais)

Os resultados consolidados da pesquisa (média ± desvio padrão entre execuções independentes) reproduzem fielmente os dados da pesquisa:

| Fase Experimental | N | Acurácia (%) | Precisão (%) | Sensibilidade (%) | F1-Score (%) | AUC-ROC | Perda de Validação |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fase 1 – MLP** | 30 | 94,87 ± 1,71 | 95,21 ± 2,58 | 94,58 ± 2,66 | 94,85 ± 1,70 | 0,9863 ± 0,0101 | 0,1447 ± 0,0457 |
| **Fase 2 – Atenção + Residual** | 30 | 94,13 ± 1,95 | 93,98 ± 3,93 | 94,58 ± 3,11 | 94,18 ± 1,85 | 0,9835 ± 0,0105 | 0,1726 ± 0,0415 |
| **Fase 3 – Base Enriquecida** | 30 | 94,01 ± 4,17 | 95,81 ± 1,86 | 92,14 ± 8,91 | 93,66 ± 5,43 | 0,9838 ± 0,0090 | 0,1787 ± 0,0569 |
| **Fase 4 – Focal Loss + LayerNorm** | 30 | 95,60 ± 0,82 | 95,79 ± 1,17 | 95,43 ± 1,43 | 95,60 ± 0,84 | 0,9858 ± 0,0047 | 0,0278 ± 0,0059 |
| **Fase 4 – Repetibilidade** | 30 | 95,64 ± 0,86 | 95,87 ± 1,06 | 95,43 ± 1,64 | 95,64 ± 0,88 | 0,9863 ± 0,0044 | 0,0280 ± 0,0062 |
| **Fase 5 – AMP (Configuração Final)** | 35 | 95,22 ± 0,98 | 95,58 ± 1,28 | 94,86 ± 1,51 | 95,21 ± 0,99 | 0,9856 ± 0,0044 | 0,0272 ± 0,0058 |

> **Conclusão Central:** O benefício primário da sofisticação arquitetural e das técnicas de regularização não foi um salto desproporcional nas médias, mas a **redução drástica da variabilidade entre execuções** (desvios padrão reduzidos pela metade em relação à Fase 1 e quase quatro vezes menores que os da Fase 3).

---

## 📂 Estrutura do Repositório

```text
├── data/
│   ├── raw/
│   │   └── dataset.csv              # Base tabular oficial
│   └── processed/
│       ├── scaler.joblib            # RobustScaler ajustado no treino
│       ├── resultados_30_experimentos.csv              # Resultados da Fase 1
│       ├── resultados_30_experimentos_aprimorados.csv   # Resultados da Fase 2
│       ├── resultados_30_experimentos_alteracao3.csv    # Resultados da Fase 3
│       ├── resultados_30_experimentos_alteracao4.csv    # Resultados da Fase 4
│       └── resultados_35_experimentos_alteracao5.csv    # Resultados da Fase 5
├── models/
│   └── best_model.pt                # Checkpoint do melhor modelo treinado
├── src/
│   ├── __init__.py
│   ├── config.py                    # Caminhos dinâmicos (pathlib), sementes e hiperparâmetros
│   ├── dataset.py                   # DataPipeline, RobustScaler e engenharia de atributos
│   ├── model.py                     # ClassifierMLP, GatedFeatureAttention, StableResidualBlock
│   ├── train.py                     # Treinamento de ciclo único com Early Stopping
│   ├── evaluate.py                  # Avaliação detalhada no teste (Acurácia, F1, ROC, CM)
│   ├── train_multiple.py            # Treinamentos com sementes fixas
│   ├── run_30_experiments.py        # Bateria da Fase 4 (30 ciclos)
│   ├── run_35_experiments.py        # Bateria da Fase 5 (35 ciclos com AMP)
│   └── main_api.py                  # API de inferência (FastAPI - Componente extra)
├── main.py                          # Script orquestrador de execução
├── requirements.txt                 # Dependências com versões fixadas
├── .env.example                     # Modelo de variáveis de ambiente
├── .gitignore                       # Filtro de segurança e integridade
├── LICENSE                          # Licença MIT
└── README.md
```

---

## 📋 Metodologia e Base de Dados

- **Especificação Metodológica do Artigo**: A base experimental é composta por instâncias numéricas padronizadas com 20 variáveis originais de entrada e uma coluna de saída binária (`target` 0 e 1). A engenharia de características gera 4 variáveis derivadas (`feat_1_x_2`, `feat_1_div_2`, `feat_3_sq` e `feat_4_abs`), totalizando **24 preditores**.
- **Divisão Estratificada**: 70% Treinamento, 15% Validação (usada para Early Stopping e calibração de hiperparâmetros) e 15% Teste (isolado para avaliação de generalização).
- **Proteção contra Vazamento de Dados (*Data Leakage*)**: O `RobustScaler` é ajustado (*fit*) exclusivamente com a partição de treino. O ruído gaussiano (σ = 0,01) é aplicado somente sobre as variáveis de treino.
- **Nota Técnica sobre a Função de Contingência (*Fallback*)**: A função `generate_synthetic_dataset` contida em `src/dataset.py` atua como rotina secundária de salvaguarda (geração sintética de emergência caso o arquivo `dataset.csv` não esteja presente no caminho especificado).

---

## ⚙️ Pré-requisitos e Instalação

- **Linguagem**: Python 3.10 ou superior (testado e homologado no Python 3.13.14).
- **Ambiente de Hardware**: CPU ou GPU compatível com CUDA.
  > *Nota sobre o AMP*: A Precisão Mista Automática (*Automatic Mixed Precision*) é acelerada em GPUs com suporte a CUDA; em CPU, o código executa com precisão padrão mantendo total compatibilidade.

### 1. Clonar o Repositório
```bash
git clone https://github.com/IsaqueTF/rede-neural-dados-tabulares.git
cd rede-neural-dados-tabulares
```

### 2. Criar e Ativar o Ambiente Virtual
**No Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**No Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```
*(Para instalar PyTorch com suporte a aceleração CUDA no Windows/Linux, consulte as instruções em [pytorch.org](https://pytorch.org/get-started/locally/))*.

---

## 🚀 Como Reproduzir os Resultados da Tabela 1

As sementes pseudo-aleatórias de cada fase seguem a fórmula determinística centralizada em `src/config.py`:  
`seeds = [11 + i * 37 for i in range(N)]`

| Fase | Comando de Execução | Arquivo de Saída Gerado | Tempo Médio Estimado |
|:---:|---|---|:---:|
| **Fase 5 (AMP)** | `python main.py --modo experimentos --fase 5` | `data/processed/resultados_35_experimentos_alteracao5.csv` | ~1 a 3 min (GPU) |
| **Fase 4 (Focal+LN)** | `python main.py --modo experimentos --fase 4` | `data/processed/resultados_30_experimentos_alteracao4.csv` | ~1 a 3 min (GPU) |
| **Fases 1 a 3** | Resultados históricos consolidados em `data/processed/` | `data/processed/resultados_30_experimentos*.csv` | — |

*Dica para teste rápido de verificação:* adicione `--ciclos 1` para rodar uma única iteração de conferência:
```bash
python main.py --modo experimentos --fase 5 --ciclos 1
```

### Treinamento Individual e Avaliação
```bash
# Treinamento de ciclo único com Early Stopping
python main.py --modo treino

# Avaliação no conjunto de teste independente (Acurácia, F1, AUC, Matriz de Confusão)
python main.py --modo avaliar
```

---

## 🌐 API de Inferência (Componente Extra)

O projeto disponibiliza uma API REST assíncrona desenvolvida em FastAPI como **componente extra** (fora do escopo estrito do artigo) para demonstrar a aplicação do modelo em ambiente de produção.

Para executar o servidor:
```bash
uvicorn src.main_api:app --host 127.0.0.1 --port 8000
```
Acesse a documentação interativa Swagger em: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔒 Aviso sobre Artefatos Binários e Segurança

Os arquivos [`models/best_model.pt`](models/best_model.pt) e [`data/processed/scaler.joblib`](data/processed/scaler.joblib) foram gerados e verificados pelos autores. Como arquivos `.pt` (PyTorch) e `.joblib` utilizam internamente o mecanismo de serialização *pickle*, eles só devem ser carregados a partir deste repositório oficial ou de fontes confiáveis.

---

## 📑 Como Citar

Se este trabalho ou código for útil em sua pesquisa, por favor cite:

```bibtex
@article{ferreira2026desenvolvimento,
  title={Desenvolvimento da Intelig\^encia Artificial: estrutura\c{c}\~ao do banco de dados, treinamento, codifica\c{c}\~ao da IA, testes e valida\c{c}\~ao},
  author={Ferreira, Isaque Teles and Borges, Igor Dalarmelino},
  journal={Universidade Evang\'elica de Goi\'as -- UniEVANG\'ELICA},
  year={2026}
}
```

Formato ABNT:
> FERREIRA, Isaque Teles; BORGES, Igor Dalarmelino. **Desenvolvimento da Inteligência Artificial: estruturação do banco de dados, treinamento, codificação da IA, testes e validação**. UniEVANGÉLICA, 2026.

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo [LICENSE](LICENSE) para obter detalhes.
