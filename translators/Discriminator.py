import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, latent_dim, discriminator_dim: int = 1024, depth: int = 3, weight_init: str = 'kaiming'):
        super().__init__()
        self.latent_dim = latent_dim

        assert depth >= 1, "Depth must be at least 1"
        self.layers = nn.ModuleList()
        if depth >= 2:
            layers = []
            layers.append(nn.Linear(latent_dim, discriminator_dim))
            layers.append(nn.Dropout(0.3))
            for _ in range(depth - 2):
                layers.append(nn.SiLU())
                layers.append(nn.Linear(discriminator_dim, discriminator_dim))
                layers.append(nn.LayerNorm(discriminator_dim))
                layers.append(nn.Dropout(0.3))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(discriminator_dim, 1))
            self.layers.append(nn.Sequential(*layers))
        else:
            layers.append(nn.Linear(latent_dim, 1))
        self.initialize_weights(weight_init)
    
    def initialize_weights(self, weight_init: str):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                elif weight_init == 'xavier':
                    torch.nn.init.xavier_normal_(module.weight)
                elif weight_init == 'orthogonal':
                    torch.nn.init.orthogonal_(module.weight)
                else:
                    raise ValueError(f"Unknown weight initialization: {weight_init}")
                module.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
