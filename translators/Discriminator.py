import torch
import torch.nn as nn
from translators.MLPWithResidual import MLPWithResidual


def add_residual(input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if input_x.shape[1] < x.shape[1]:
        padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device)
        input_x = torch.cat([input_x, padding], dim=1)
    elif input_x.shape[1] > x.shape[1]:
        input_x = input_x[:, :x.shape[1]]
    return x + input_x


class Discriminator(nn.Module):
    def __init__(self, latent_dim, discriminator_dim: int = 1024, depth: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        assert depth >= 1, "Depth must be at least 1"
        self.layers = nn.ModuleList()
        if depth >= 2:
            layers = []
            layers.append(nn.Linear(latent_dim, discriminator_dim))
            for _ in range(depth - 2):
                layers.append(nn.SiLU())
                layers.append(nn.Linear(discriminator_dim, discriminator_dim))
                layers.append(nn.LayerNorm(discriminator_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(discriminator_dim, 1))
            self.layers.append(nn.Sequential(*layers))
        else:
            layers.append(nn.Linear(latent_dim, 1))
        self.initialize_weights()
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                module.bias.data.fill_(0)

    def forward(self, x):
        input_x = x
        for layer in self.layers:
            x = layer(x)
            # x = add_residual(input_x, x)
            input_x = x
        return x
