import torch.nn as nn
from translators.MLPWithResidual import MLPWithResidual

class Discriminator(nn.Module):
    def __init__(self, latent_dim, discriminator_dim=1024, depth=3, use_residual=False, norm_style=None):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim

        if use_residual:
            self.discriminator = MLPWithResidual(depth, latent_dim, discriminator_dim, 1, norm_style)
        else:
            assert depth >= 1
            layers = []
            layers.append(nn.Linear(latent_dim, discriminator_dim))
            for _ in range(depth - 1):
                layers.append(nn.Tanh())
                layers.append(nn.Linear(discriminator_dim, discriminator_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Linear(discriminator_dim, 1))
            self.discriminator = nn.Sequential(*layers)


    def forward(self, x):
        return self.discriminator(x)