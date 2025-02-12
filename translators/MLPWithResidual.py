import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


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
            use_spectral_norm: bool = True,
            output_norm: bool = False,
        ):
        super().__init__()
        self.depth = depth
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layers = nn.ModuleList()

        spec = spectral_norm if use_spectral_norm else lambda x: x

        if norm_style == 'batch':
            norm_layer = nn.BatchNorm1d
        elif norm_style == 'layer':
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm style: {norm_style}")


        for layer_idx in range(self.depth):
            ################################################################
            if layer_idx == 0:
                hidden_dim = out_dim if self.depth == 1 else hidden_dim
                self.layers.append(
                    nn.Sequential(
                        spec(nn.Linear(in_dim, hidden_dim)),
                        nn.SiLU(),
                        # norm_layer(hidden_dim),
                    )
                )
            elif layer_idx < self.depth - 1:
                self.layers.append(
                    nn.Sequential(
                        spec(nn.Linear(hidden_dim, hidden_dim)),
                        nn.SiLU(),
                        norm_layer(hidden_dim),
                        nn.Dropout(p=0.1),
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        spec(nn.Linear(hidden_dim, hidden_dim)),
                        nn.Dropout(p=0.1),
                        nn.SiLU(),
                        spec(nn.Linear(hidden_dim, out_dim)),
                    )
                )
        self.initialize_weights()
        if output_norm:
            self.output_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        else:
            self.output_norm = None
    
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
        
        if self.output_norm is not None:
            x = self.output_norm(x)

        return x