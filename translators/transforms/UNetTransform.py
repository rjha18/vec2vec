from transformers import AutoModelForMaskedLM

import diffusers
import torch
from translators.transforms.AbsTransform import AbsTransform


class UNetTransform(AbsTransform):
    def __init__(self, src_dim: int, target_dim: int):
        super().__init__()
        
        self.base = diffusers.UNet2DModel(
            sample_size=(src_dim // 32, 32), # the target image resolution
            in_channels=1, # the number of input channels, 3 for RGB images
            out_channels=1, # the number of output channels
            layers_per_block=2, # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512), # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D", # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D", # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", # a regular ResNet upsampling block
                "AttnUpBlock2D", # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.internal_dim = src_dim
        self.reshape_factor = 32
        assert self.internal_dim % self.reshape_factor == 0
        self.project = torch.nn.Linear(src_dim, target_dim)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = x[:, None, None, :].reshape(-1, 1, self.internal_dim // self.reshape_factor, self.reshape_factor) # repeat embedding into an image
        output = self.base(x, timestep=0)
        z = output.sample.reshape(batch_size, self.internal_dim)
        return self.project(z)
