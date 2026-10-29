"""
Arquitetura Neural Alteracao 4:
- Feature Attention calibrada
- Residual Blocks com LayerNorm e SiLU
- Skip-connections com projecao residual balanceada
"""
import torch
import torch.nn as nn

class GatedFeatureAttention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class StableResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.10):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + 0.5 * self.block(x))


class ClassifierMLP(nn.Module):
    """
    MLP Robusto com Gated Attention, LayerNorm e Regularizacao Balanceada.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.10):
        super().__init__()
        
        self.attention = GatedFeatureAttention(input_dim)
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.res1 = StableResidualBlock(hidden_dim, dropout_rate)
        self.res2 = StableResidualBlock(hidden_dim, dropout_rate)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_att = self.attention(x)
        x_proj = self.input_layer(x_att)
        x_res = self.res1(x_proj)
        x_res = self.res2(x_res)
        return self.head(x_res)