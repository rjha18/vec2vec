import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spec


def add_residual(input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if input_x.shape[1] < x.shape[1]:
        padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device)
        input_x = torch.cat([input_x, padding], dim=1)
    elif input_x.shape[1] > x.shape[1]:
        input_x = input_x[:, :x.shape[1]]
    return x + input_x


class MLPWithResidual(nn.Module):
    def __init__(
            self, 
            depth: int, 
            in_dim: int, 
            hidden_dim: int, 
            out_dim: int,
            norm_style: bool = 'batch',
        ):
        super().__init__()
        self.depth = depth
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layers = nn.ModuleList()

        assert norm_style in ['batch', 'layer', 'spectral', None]
        batch_norm = None
        if norm_style == 'batch':
            batch_norm = nn.BatchNorm1d
        elif norm_style == 'layer':
            batch_norm = nn.LayerNorm


        for layer_idx in range(self.depth):
            ################################################################
            if layer_idx == 0:
                hidden_dim = out_dim if self.depth == 1 else hidden_dim
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, hidden_dim) if norm_style != 'spectral' else spec(nn.Linear(in_dim, hidden_dim)),
                        nn.SiLU(),
                        nn.Dropout(p=0.01),
                        batch_norm(hidden_dim) if batch_norm is not None else nn.Identity(),
                    )
                )
            elif layer_idx < self.depth - 1:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim) if norm_style != 'spectral' else spec(nn.Linear(hidden_dim, hidden_dim)),
                        nn.SiLU(),
                        nn.Dropout(p=0.01),
                        batch_norm(hidden_dim) if batch_norm is not None else nn.Identity(),
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim) if norm_style != 'spectral' else spec(nn.Linear(hidden_dim, hidden_dim)),
                        nn.Dropout(p=0.01),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, out_dim) if norm_style != 'spectral' else spec(nn.Linear(hidden_dim, out_dim)),
                    )
                )
        self.initialize_weights()
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
         
    def forward(self, x):
        for layer in self.layers:
            input_x = x
            x = layer(x)
            x = add_residual(input_x, x)
        
        return x