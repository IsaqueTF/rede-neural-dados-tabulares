from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib
import torch

from src import config
from src.model import ClassifierMLP

# Gerenciamento de estado dos artefatos
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    scaler_path = config.PROCESSED_DATA_DIR / "scaler.joblib"
    model_path = config.MODELS_DIR / "best_model.pt"

    if not scaler_path.exists() or not model_path.exists():
        raise RuntimeError("Artefatos de modelo ou scaler não encontrados. Execute train.py primeiro.")

    scaler = joblib.load(scaler_path)
    input_dim = len(scaler.mean_)

    model = ClassifierMLP(input_dim=input_dim)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    
    # Suporte para checkpoints gravados com dict ou state_dict puro
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    ml_models["model"] = model
    ml_models["scaler"] = scaler
    yield
    ml_models.clear()

app = FastAPI(title="AI Inference API", version="1.1.0", lifespan=lifespan)

class InferenceRequest(BaseModel):
    features: list[float] = Field(..., description="Vetor numérico de features de entrada")

class InferenceResponse(BaseModel):
    probability: float
    predicted_class: int

@app.post("/predict", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
def predict(request: InferenceRequest):
    scaler = ml_models["scaler"]
    model = ml_models["model"]

    if len(request.features) != len(scaler.mean_):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Features inválidas. Esperado: {len(scaler.mean_)}, Recebido: {len(request.features)}"
        )

    try:
        scaled_feats = scaler.transform([request.features])
        tensor_input = torch.tensor(scaled_feats, dtype=torch.float32)

        with torch.inference_mode():
            logit = model(tensor_input)
            prob = torch.sigmoid(logit).item()
            pred_class = int(prob >= 0.5)

        return InferenceResponse(probability=round(prob, 4), predicted_class=pred_class)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))