from pathlib import Path
import random
import numpy as np
import torch

# Diretorios Raiz (portabilidade com pathlib)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Garante a criacao dos diretorios
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Caminhos dos Artefatos
MODEL_PATH = MODELS_DIR / "best_model.pt"
SCALER_PATH = PROCESSED_DATA_DIR / "scaler.joblib"

# Hiperparametros Centrais
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 7

# Dispositivo de Processamento
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = RANDOM_STATE):
    """Fixa sementes para random, numpy e torch (incluindo CUDA) garantindo reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
