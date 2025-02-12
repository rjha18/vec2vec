import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spec
from translators.MLPWithResidual import MLPWithResidual

class Discriminator(nn.Module):
    def __init__(self, latent_dim, discriminator_dim=1024, depth=3):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim

        assert depth >= 1, "Depth must be at least 1"
        layers = []
        if depth >= 2:
            layers.append(spec(nn.Linear(latent_dim, discriminator_dim)))
            for _ in range(depth - 2):
                layers.append(nn.SiLU())
                layers.append(spec(nn.Linear(discriminator_dim, discriminator_dim)))
            layers.append(nn.SiLU())
            layers.append(spec(nn.Linear(discriminator_dim, 1)))
        else:
            layers.append(spec(nn.Linear(latent_dim, 1)))
        self.discriminator = nn.Sequential(*layers)
        self.initialize_weights()
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        output = self.discriminator(x)
        # breakpoint()
        return output