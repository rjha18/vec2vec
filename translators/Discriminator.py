import torch
import torch.nn as nn

from translators.MLPMixer import MLPMixer
from translators.MLPWithResidual import MLPWithResidual


class Discriminator(nn.Module):
    def __init__(
            self, 
            cfg, 
            latent_dim: int, 
            discriminator_dim: int = 1024, 
            depth: int = 3, 
            weight_init: str = "kaiming"
        ):
        super().__init__()
        self.latent_dim = latent_dim
        if cfg.style == 'res_mlp':
            self.backbone = MLPWithResidual(
                depth=depth,
                in_dim=latent_dim,
                hidden_dim=discriminator_dim, 
                out_dim=latent_dim, 
                norm_style="layer",
                weight_init=weight_init,
            )
        elif cfg.style == "mixer":
            self.backbone = MLPMixer(
                depth=cfg.transform_depth,
                in_dim=latent_dim,
                hidden_dim=discriminator_dim,
                out_dim=latent_dim,
                num_patches=cfg.mixer_num_patches,
                weight_init=weight_init,
            )
        self.output = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, 1)
       )

    def forward(self, x):
        x = self.backbone(x)
        return self.output(x)
