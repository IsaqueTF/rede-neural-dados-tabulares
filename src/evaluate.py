"""
Modulo de avaliacao rigorosa do modelo no conjunto de teste.
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src import config
from src.dataset import DataPipeline
from src.model import ClassifierMLP

def evaluate():
    pipeline = DataPipeline(config.RAW_DATA_PATH, config.PROCESSED_DATA_DIR, random_state=config.RANDOM_STATE)
    _, _, test_loader, input_dim = pipeline.get_dataloaders(target_column="target", batch_size=config.BATCH_SIZE)

    model = ClassifierMLP(input_dim=input_dim).to(config.DEVICE)
    
    # Carregamento seguro do checkpoint usando config.MODEL_PATH
    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(config.DEVICE)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            y_probs.extend(probs)
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy().flatten())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 45)
    print("      RELATORIO DE AVALIACAO DE TESTE        ")
    print("=" * 45)
    print(f"Acuracia:         {acc:.4f}")
    print(f"Precisao:         {prec:.4f}")
    print(f"Recall:           {rec:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"AUC-ROC:          {auc:.4f}")
    print("\nMatriz de Confusao:")
    print(cm)
    print("=" * 45 + "\n")

if __name__ == "__main__":
    evaluate()