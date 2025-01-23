import torch
import random
from torch import nn
from translators.AbsNTranslator import AbsNTranslator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearTranslator(AbsNTranslator):
    def __init__(
        self,
        encoder_dims: dict[str, int],
        normalize_embeddings: bool = True,
        src_emb: str = None,
        tgt_emb: str = None,
    ):
        if src_emb is not None and tgt_emb is not None:
            assert src_emb in encoder_dims
            assert tgt_emb in encoder_dims
            assert encoder_dims[src_emb] == encoder_dims[tgt_emb]
            src_dim = encoder_dims[src_emb]
            tgt_dim = encoder_dims[tgt_emb]
        else:
            assert len(encoder_dims) == 1
            dims = list(set(encoder_dims.values()))
            src_dim, tgt_dim = dims[0]
        super().__init__(encoder_dims, -1, -1)

        self.translator = nn.Sequential(nn.Linear(src_dim, tgt_dim))
        self.normalize_embeddings = normalize_embeddings
        self.style = 'linear'

    def _make_adapters(self):
        return
    
    def translate_embeddings(
            self, embeddings: torch.Tensor, in_name: str, out_name: str, max_noise_pow: float = 0, min_noise_pow: float = 0
    ) -> torch.Tensor:
        return self._translate(embeddings)
    
    def forward(self, ins: dict[str, torch.Tensor], max_noise_pow: float, min_noise_pow: float) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        recons = {}
        translations = {
            flag: {} for flag in ins.keys()
        }

        for flag, emb in ins.items():
            if self.training and not (max_noise_pow == 0 and min_noise_pow == 0):
                noise_pow = random.uniform(min_noise_pow, max_noise_pow)
                noise_level = 10 ** noise_pow
                noise = torch.randn_like(emb) * noise_level
            else:
                noise = torch.zeros_like(emb)
            out = self._translate(emb + noise)
            for target_flag in ins.keys():
                if target_flag == flag:
                    recons[flag] = out
                else:
                    translations[target_flag][flag] = out
        
        return recons, translations

    def _translate(self, embs: torch.Tensor) -> torch.Tensor:
        out = self.translator(embs)

        if self.normalize_embeddings:
            out = out / out.norm(dim=1, keepdim=True)
        return out