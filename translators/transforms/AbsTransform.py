from torch import nn
from abc import ABC, abstractmethod


class AbsTransform(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self
    ):
        super().__init__()
        self.transform = None

    def forward(self, x):
        return self.transform(x)