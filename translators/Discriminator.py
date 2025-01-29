import torch.nn as nn
from translators.MLPWithResidual import MLPWithResidual

class Discriminator(nn.Module):
    def __init__(self, latent_dim, discriminator_dim=1024, depth=3, use_residual=False):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim

        self.input = nn.Linear(latent_dim, discriminator_dim)

        if use_residual:
            self.disc_layers = MLPWithResidual(depth, discriminator_dim, discriminator_dim, discriminator_dim)
        else:
            layers = []
            for _ in range(depth):
                layers.append(nn.Tanh())
                layers.append(nn.Linear(discriminator_dim, discriminator_dim))
            layers.append(nn.Tanh())
            self.disc_layers = nn.Sequential(*layers)

        self.output = nn.Linear(discriminator_dim, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.disc_layers(x)
        return self.output(x)