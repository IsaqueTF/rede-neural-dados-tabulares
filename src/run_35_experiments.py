"""
Bateria de 35 Treinamentos - Alteracao 5:
- 35 ciclos independentes
- Otimizacao de memoria (gc.collect e torch.cuda.empty_cache)
- Precisao Mista Automatica (AMP) para menor consumo de VRAM e maior velocidade
- Exportacao em 'resultados_35_experimentos_alteracao5.csv'
"""
import argparse
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src import config
from src.dataset import DataPipeline
from src.model import ClassifierMLP

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        return (focal_weight * bce_loss).mean()

def run_single_cycle(seed: int, epochs: int = config.EPOCHS, lr: float = 1.5e-3, batch_size: int = config.BATCH_SIZE):
    pipeline = DataPipeline(config.RAW_DATA_PATH, config.PROCESSED_DATA_DIR, random_state=seed)
    train_loader, val_loader, test_loader, input_dim = pipeline.get_dataloaders(
        target_column="target", 
        batch_size=batch_size
    )

    config.set_seed(seed)

    model = ClassifierMLP(input_dim=input_dim).to(config.DEVICE)
    criterion = BinaryFocalLoss(gamma=1.5, alpha=0.5)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda" if config.DEVICE.type == "cuda" else "cpu", enabled=(config.DEVICE.type == "cuda"))

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(config.DEVICE, non_blocking=True), y_b.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=config.DEVICE.type, enabled=(config.DEVICE.type == "cuda")):
                out = model(X_b)
                loss = criterion(out, y_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # Validacao
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(config.DEVICE, non_blocking=True), y_b.to(config.DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=config.DEVICE.type, enabled=(config.DEVICE.type == "cuda")):
                    out = model(X_b)
                    val_loss += criterion(out, y_b).item() * X_b.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 12 and epoch >= 15:
                break

    # Avaliacao de Teste
    model.load_state_dict(best_weights)
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(config.DEVICE)
            logits = model(X_b)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            y_probs.extend(probs)
            y_pred.extend(preds)
            y_true.extend(y_b.numpy().flatten())

    # Liberacao de memoria entre ciclos
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "ciclo": None,
        "seed": seed,
        "stopped_epoch": epoch,
        "best_val_loss": round(best_val_loss, 4),
        "acuracia": round(accuracy_score(y_true, y_pred), 4),
        "precisao": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_true, y_probs), 4)
    }

def main(total_cycles: int = 35):
    parser = argparse.ArgumentParser(description="Execução da Bateria Fase 5 (AMP e Otimização)")
    parser.add_argument("--ciclos", type=int, default=total_cycles, help="Número de ciclos independentes (padrão: 35)")
    args, _ = parser.parse_known_args()
    total_cycles = args.ciclos

    seeds = [i * 37 + 11 for i in range(total_cycles)]
    records = []

    print("=" * 70)
    print(f" INICIANDO BATERIA DE {total_cycles} TESTES - FASE 5 (AMP E OTIMIZACAO) ")
    print("=" * 70)

    for idx, s in enumerate(seeds, start=1):
        print(f"Executando Ciclo [{idx:02d}/{total_cycles}] | Seed: {s}...")
        res = run_single_cycle(seed=s)
        res["ciclo"] = idx
        records.append(res)

    df = pd.DataFrame(records)
    cols = ["ciclo", "seed", "stopped_epoch", "best_val_loss", "acuracia", "precisao", "recall", "f1_score", "auc_roc"]
    df = df[cols]

    csv_path = config.PROCESSED_DATA_DIR / "resultados_35_experimentos_alteracao5.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print(f"      {total_cycles} EXPERIMENTOS CONCLUIDOS COM SUCESSO (FASE 5)       ")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nArquivo salvo em: {csv_path}")

if __name__ == "__main__":
    main()
