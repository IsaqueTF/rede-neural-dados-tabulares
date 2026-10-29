"""
Script para execucao automatizada de multiplos ciclos de treinamento.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from src import config
from src.dataset import DataPipeline
from src.model import ClassifierMLP

def run_experiment(seed: int, epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
    # 1. Pipeline com semente variavel para cada rodada
    pipeline = DataPipeline(config.RAW_DATA_PATH, config.PROCESSED_DATA_DIR, random_state=seed)
    train_loader, val_loader, _, input_dim = pipeline.get_dataloaders(target_column="target", batch_size=batch_size)

    # 2. Inicializacao do Modelo e Otimizador
    torch.manual_seed(seed)
    model = ClassifierMLP(input_dim=input_dim).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Fase de Treino
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Fase de Validacao
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                break

    return {
        "seed": seed,
        "final_train_loss": round(epoch_train_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "stopped_epoch": epoch
    }

def main():
    # Lista de sementes para executar 5 treinamentos independentes
    seeds = [10, 42, 100, 2024, 777]
    results = []

    print(f"Iniciando bateria de {len(seeds)} treinamentos...")

    for i, seed in enumerate(seeds, start=1):
        print(f"Executando Ciclo [{i}/{len(seeds)}] com Seed={seed}...")
        res = run_experiment(seed=seed)
        results.append(res)

    # Consolidacao em DataFrame
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 50)
    print("        RESULTADOS DOS MULTIPLOS TREINAMENTOS        ")
    print("=" * 50)
    print(df_results.to_string(index=False))
    print("=" * 50)
    print(f"Media da Melhor Perda de Validacao: {df_results['best_val_loss'].mean():.4f}")
    print(f"Desvio Padrao da Perda de Validacao: {df_results['best_val_loss'].std():.4f}")

if __name__ == "__main__":
    main()