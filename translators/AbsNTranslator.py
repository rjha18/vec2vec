from abc import ABC, abstractmethod
from huggingface_hub import PyTorchModelHubMixin

import torch
from torch import nn

class AbsNTranslator(nn.Module, ABC, PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_dims: dict[str, int],
        d_adapter: int,
        depth: int = 3,
    ):
        super().__init__()
        if d_adapter is None:
            d_adapter = d_model
        
        self.n = len(encoder_dims)
        self.d_adapter = d_adapter
        self.depth = depth
        self.in_adapters = nn.ModuleDict()
        self.out_adapters = nn.ModuleDict()
        self.transform = None

    @abstractmethod
    def _make_adapters(self):
        pass

    @abstractmethod
    def forward(self, ins: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pass
