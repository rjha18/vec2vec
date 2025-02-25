from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


def add_residual(input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if input_x.shape[1] < x.shape[1]:
        padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device)
        input_x = torch.cat([input_x, padding], dim=1)
    elif input_x.shape[1] > x.shape[1]:
        input_x = input_x[:, :x.shape[1]]
    return x + input_x


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
        self.initialize_weights(weight_init)
        self.output = nn.Linear(hidden_dim, out_dim)
    
    def initialize_weights(self, weight_init: str):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                print("initializing", type(module))
                if weight_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                elif weight_init == 'xavier':
                    torch.nn.init.xavier_normal_(module.weight)
                elif weight_init == 'orthogonal':
                    torch.nn.init.orthogonal_(module.weight)
                else:
                    raise ValueError(f"Unknown weight initialization: {weight_init}")
                module.bias.data.fill_(0)
            elif isinstance(module, nn.BatchNorm1d):
                torch.nn.init.normal_(module.weight, mean=1.0, std=0.02)
                torch.nn.init.normal_(module.bias, mean=0.0, std=0.02) 
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.constant_(module.weight, 1.0)
         
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x[:, None, :])
        x = self.mixer_blocks(x)
        x = x.mean(dim=1) # global mean pooling
        return self.output(x)
