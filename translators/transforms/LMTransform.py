from transformers import AutoModelForMaskedLM
import torch
from translators.transforms.AbsTransform import AbsTransform


class LMTransform(AbsTransform):
    def __init__(
            self,
            src_dim: int,
            target_dim: int,
            base_name: str,
            scale: int = 16
        ):
        super().__init__()

        self.scale = scale
        self.base = AutoModelForMaskedLM.from_pretrained(base_name)
        self.internal_dim = self.base.base_model.embeddings.word_embeddings.embedding_dim

        self.model_dim = self.internal_dim * self.scale
        self.upscale = torch.nn.Linear(src_dim, self.model_dim)
        self.downscale = torch.nn.Linear(self.model_dim, target_dim)

    def forward(self, x: torch.Tensor):
        z = self.upscale(x).reshape(-1, self.scale, self.internal_dim)
        z = self.base(inputs_embeds=z, output_hidden_states=True)
        return self.downscale(z.hidden_states[-1].reshape(-1, self.model_dim))
