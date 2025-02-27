from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


class EmbPatches(nn.Module):
    def __init__(self, input_dim: int, num_patches: int, hidden_dim: int):
        super().__init__()
        assert input_dim % num_patches == 0
        self.num_patches = num_patches
        self.patch_size = input_dim // num_patches
        self.patch_weights = nn.Parameter(torch.randn(self.patch_size, hidden_dim))
        self.projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        patches = emb.reshape(emb.shape[0], self.num_patches, self.patch_size)
        patches = torch.einsum('bpd, dh -> bph', patches, self.patch_weights)
        proj = self.projection(patches)
        return proj


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, num_patches, chn_dim, tok_hid_dim, chn_hid_dim, p=0.):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(chn_dim),
            Rearrange('b t d -> b d t'),
            MLPBlock(num_patches, tok_hid_dim, p),
            Rearrange('b d t -> b t d')
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(chn_dim),
            MLPBlock(chn_dim, chn_hid_dim, p)
        )
        
    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class MLPMixer(nn.Module):
    def __init__(
            self, 
            depth: int, 
            in_dim: int, 
            hidden_dim: int, 
            out_dim: int,
            num_patches: int,
            weight_init: str = 'kaiming',
            p: float = 0.1,
        ):
        super().__init__()
        self.depth = depth
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layers = nn.ModuleList()

        self.patch = EmbPatches(
            input_dim=in_dim, 
            num_patches=num_patches, 
            hidden_dim=hidden_dim,
        )
        self.mixer_blocks = nn.Sequential(*[
            MixerBlock(
                num_patches=num_patches, 
                chn_dim=hidden_dim, 
                tok_hid_dim=hidden_dim, 
                chn_hid_dim=hidden_dim,
                p=p,
            )  for _ in range(depth)
        ])

        assert out_dim % num_patches == 0, f"Output dim {out_dim} must be divisible by num_patches {num_patches}"
        self.downproject = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim // num_patches),
            Rearrange('b t d -> b (t d)'),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x[:, None, :])
        x = self.mixer_blocks(x)
        return self.downproject(x)

