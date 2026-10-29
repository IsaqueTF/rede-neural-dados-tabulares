"""
Pipeline de Dados Aprimorado com Engenharia de Atributos e RobustScaler.
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from src import config

class DataPipeline:
    def __init__(self, raw_path: Path, processed_dir: Path, random_state: int = config.RANDOM_STATE):
        self.raw_path = Path(raw_path)
        self.processed_dir = Path(processed_dir)
        self.random_state = random_state
        self.scaler = RobustScaler()

    def generate_synthetic_dataset(self, n_samples: int = 5000, n_features: int = 15) -> pd.DataFrame:
        """Gera um conjunto sintetico rico com caracteristicas informativas, redundantes e repetidas."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=3,
            n_repeated=0,
            n_classes=2,
            weights=[0.5, 0.5],
            flip_y=0.01,  # 1% de ruido de rotulo realista
            class_sep=1.2,
            random_state=self.random_state
        )
        
        feature_cols = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = y
        return df

    def _engineer_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Gera interacoes polinomiais e combinacoes nao-lineares entre variaveis."""
        df_feat = df.copy()
        
        # Interacoes entre variaveis informativas de topo
        if "feature_1" in df_feat.columns and "feature_2" in df_feat.columns:
            df_feat["feat_1_x_2"] = df_feat["feature_1"] * df_feat["feature_2"]
            df_feat["feat_1_div_2"] = df_feat["feature_1"] / (df_feat["feature_2"].abs() + 1e-5)
            
        if "feature_3" in df_feat.columns and "feature_4" in df_feat.columns:
            df_feat["feat_3_sq"] = np.sin(df_feat["feature_3"])
            df_feat["feat_4_abs"] = np.abs(df_feat["feature_4"])
            
        return df_feat

    def load_data(self, target_column: str = "target") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carrega dados existentes ou gera a base aprimorada, particionando em 70/15/15."""
        if self.raw_path.exists():
            df = pd.read_csv(self.raw_path)
        else:
            print("Dataset bruto nao encontrado. Gerando base aprimorada de 5.000 amostras...")
            df = self.generate_synthetic_dataset(n_samples=5000, n_features=15)
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.raw_path, index=False)

        # Aplicacao de Engenharia de Recursos
        df_eng = self._engineer_features(df, target_column)

        # Divisao Estratificada: 70% Treino, 15% Validacao, 15% Teste
        train_df, temp_df = train_test_split(
            df_eng,
            test_size=0.30,
            random_state=self.random_state,
            stratify=df_eng[target_column]
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=self.random_state,
            stratify=temp_df[target_column]
        )

        return train_df, val_df, test_df

    def get_dataloaders(self, target_column: str = "target", batch_size: int = config.BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        train_df, val_df, test_df = self.load_data(target_column=target_column)

        feature_cols = [c for c in train_df.columns if c != target_column]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_column].values.astype(np.float32)

        X_val = val_df[feature_cols].values
        y_val = val_df[target_column].values.astype(np.float32)

        X_test = test_df[feature_cols].values
        y_test = test_df[target_column].values.astype(np.float32)

        # Ajuste estrito do RobustScaler no conjunto de treino
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Data Augmentation: Ruido Gaussiano leve no treino
        noise = np.random.normal(0, 0.01, X_train_scaled.shape)
        X_train_scaled += noise

        # Serializacao do escalonador
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, config.SCALER_PATH)

        # Tensores PyTorch
        t_train_x = torch.tensor(X_train_scaled, dtype=torch.float32)
        t_train_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        t_val_x = torch.tensor(X_val_scaled, dtype=torch.float32)
        t_val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        t_test_x = torch.tensor(X_test_scaled, dtype=torch.float32)
        t_test_y = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(TensorDataset(t_train_x, t_train_y), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(t_val_x, t_val_y), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(t_test_x, t_test_y), batch_size=batch_size, shuffle=False)

        input_dim = X_train_scaled.shape[1]
        return train_loader, val_loader, test_loader, input_dim