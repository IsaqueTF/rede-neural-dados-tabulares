"""
Treinamento Otimizado com Label Smoothing e Cosine Annealing with Warm Restarts.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src import config
from src.dataset import DataPipeline
from src.model import ClassifierMLP

def train():
    # 1. Pipeline de dados
    pipeline = DataPipeline(config.RAW_DATA_PATH, config.PROCESSED_DATA_DIR, random_state=config.RANDOM_STATE)
    train_loader, val_loader, _, input_dim = pipeline.get_dataloaders(
        target_column="target", 
        batch_size=config.BATCH_SIZE
    )

    # 2. Inicializacao do modelo
    torch.manual_seed(config.RANDOM_STATE)
    model = ClassifierMLP(input_dim=input_dim).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    
    # Otimizador AdamW calibrado
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-3)
    
    # Cosine Annealing: reinicia a taxa periodicamente para escapar de minimos locais
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Treinando arquitetura com Atencao Tabular em {config.DEVICE} por {config.EPOCHS} epocas...")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            
            # Label Smoothing: suaviza alvos [0, 1] para [0.05, 0.95]
            smooth_y = y_batch * 0.90 + 0.05
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, smooth_y)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Validacao (com alvos reais sem suavizacao)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # Passo do Cosine Annealing
        scheduler.step()

        print(f"Epoca [{epoch:02d}/{config.EPOCHS}] - Perda Treino: {epoch_train_loss:.4f} | Perda Validacao: {epoch_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping e salvamento do modelo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early Stopping acionado na epoca {epoch}.")
                break

    print(f"Treinamento finalizado. Melhor Perda de Validacao: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()