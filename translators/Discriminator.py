import torch.nn as nn
import torch.nn.utils.spectral_norm as spec
from translators.MLPWithResidual import MLPWithResidual

class Discriminator(nn.Module):
    def __init__(self, latent_dim, discriminator_dim=1024, depth=3):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim

        assert depth >= 1, "Depth must be at least 1"
        layers = []
        layers.append(spec(nn.Linear(latent_dim, discriminator_dim)))
        for _ in range(depth - 1):
            layers.append(nn.SiLU())
            layers.append(spec(nn.Linear(discriminator_dim, discriminator_dim)))
        layers.append(nn.SiLU())
        layers.append(spec(nn.Linear(discriminator_dim, 1)))
        self.discriminator = nn.Sequential(*layers)


    def forward(self, x):
        return self.discriminator(x)