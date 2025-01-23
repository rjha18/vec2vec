import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, latent_dim, discriminator_dim=1024, depth=3):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim

        self.input = nn.Linear(latent_dim, discriminator_dim)

        layers = []
        for _ in range(depth):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(discriminator_dim, discriminator_dim))
        layers.append(nn.ReLU())
        self.disc_layers = nn.Sequential(*layers)

        self.output = nn.Linear(discriminator_dim, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.disc_layers(x)
        return self.output(x)
