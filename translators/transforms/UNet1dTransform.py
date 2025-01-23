import diffusers
import torch

from translators.transforms.AbsTransform import AbsTransform


class UNet1dTransform(AbsTransform):
    def __init__(self, src_dim: int, target_dim: int):
        super().__init__()

        self.base = diffusers.UNet1DModel(
            sample_size=src_dim,  # the target audio length
            in_channels=1,  # the number of input channels, 
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock1D",  # a regular ResNet downsampling block
                "DownBlock1D",
                "DownBlock1D",
                "DownBlock1D",
                "AttnDownBlock1D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock1D",
            ),
            up_block_types=(
                "UpBlock1D",  # a regular ResNet upsampling block
                "AttnUpBlock1D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock1D",
                "UpBlock1D",
                "UpBlock1D",
                "UpBlock1D",
            ),
        )
        self.internal_dim = src_dim
        self.project = torch.nn.Linear(src_dim, target_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.internal_dim), f"invalid shapes: {x.shape} should be {(batch_size, self.internal_dim)}"
        
        x_r = x.view(batch_size, 1, self.internal_dim).contiguous()
        output = self.base(x_r, timestep=0)

        z = output.sample.view(batch_size, self.internal_dim).contiguous()
        return self.project(z)