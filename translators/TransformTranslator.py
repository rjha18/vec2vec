import torch
import random
from torch import nn
from translators.MLPWithResidual import MLPWithResidual
from translators.AbsNTranslator import AbsNTranslator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformTranslator(AbsNTranslator):
    def __init__(
        self,
        encoder_dims: dict[str, int],
        d_adapter: int,
        d_hidden: int,
        transform: nn.Module,
        depth: int = 3,
        normalize_embeddings: bool = True,
        style: str = 'unet',
        use_small_output_adapters: bool = False,
        use_residual_adapters: bool = False,
        norm_style: str = 'batch',
    ):
        super().__init__(encoder_dims, d_adapter, depth)

        self.d_hidden = d_hidden
        self.use_small_output_adapters = use_small_output_adapters
        self.use_residual_adapters = use_residual_adapters
        self.norm_style = norm_style
        self.transform = transform
        for flag, dims in encoder_dims.items():
            in_adapter, out_adapter = self._make_adapters(dims)
            self.in_adapters[flag] = in_adapter
            self.out_adapters[flag] = out_adapter
        self.normalize_embeddings = normalize_embeddings
        self.style = style

    def translate_embeddings(
        self, embeddings: torch.Tensor, in_name: str, out_name: str,
    ) -> torch.Tensor:
        in_adapter = self.in_adapters[in_name]
        latents = self._get_latents(emb=embeddings, in_adapter=in_adapter)
        return self._out_project(latents,  self.out_adapters[out_name])

    def add_encoders(self, encoder_dims: dict[str, int], overwrite_embs: list[str] = None):
        for flag, dims in encoder_dims.items():
            if flag in self.in_adapters and (overwrite_embs is None or flag not in overwrite_embs):
                print(f"Skipping {flag}...")
                continue
            in_adapter, out_adapter = self._make_adapters(dims)
            self.in_adapters[flag] = in_adapter
            self.out_adapters[flag] = out_adapter

    def _make_adapters(self, dims):
        assert dims is not None
        if self.use_residual_adapters:
            return (
                MLPWithResidual(self.depth, dims, self.d_hidden, self.transform.in_dim, self.norm_style),
                MLPWithResidual(self.depth, self.d_adapter, self.d_hidden, dims, self.norm_style)
            )
        in_adapter = []
        out_adapter = []
        for _ in range(self.depth):
            ################################################################
            in_adapter.append(nn.Tanh())
            in_adapter.append(nn.Linear(self.d_adapter, self.d_adapter))
            in_adapter.append(nn.BatchNorm1d(self.d_adapter))
            ################################################################
            out_adapter.append(nn.Tanh())
            out_adapter.append(nn.Linear(self.d_adapter, self.d_adapter))
            ################################################################
        in_adapter = [nn.Linear(dims, self.d_adapter)] + in_adapter

        if self.use_small_output_adapters: # backwards compatibility
            print("NOTE: Using old output adapter code for finetuning!")
            out_adapter = out_adapter[:-1]
        else:
            out_adapter.reverse()
        out_adapter = out_adapter + [nn.Linear(self.d_adapter, dims)]
        return nn.Sequential(*in_adapter), nn.Sequential(*out_adapter)

    def forward(
        self,
        ins: dict[str, torch.Tensor],
        in_set: set[str] = None,
        out_set: set[str] = None,
        include_reps: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        in_set = in_set if in_set is not None else ins.keys()
        out_set = out_set if out_set is not None else ins.keys()

        recons = {}
        translations = {
            flag: {} for flag in out_set
        }
        reps = recons.copy()

        for flag in in_set:
            noisy_emb = ins[flag]
            noisy_rep = self._get_latents(noisy_emb, self.in_adapters[flag])
            if include_reps:
                reps[flag] = noisy_rep
            for target_flag in out_set:
                if target_flag == flag:
                    recons[flag] = self._out_project(noisy_rep, self.out_adapters[flag])
                else:
                    translations[target_flag][flag] = self._out_project(noisy_rep, self.out_adapters[target_flag])

        if include_reps:
            return recons, translations, reps
        else:
            return recons, translations

    def _get_latents(self, emb: torch.Tensor, in_adapter: nn.Module) -> torch.Tensor:
        z = in_adapter(emb)
        return self.transform(z)

    def _out_project(self, emb: torch.Tensor, out_adapter: nn.Module) -> torch.Tensor:
        out = out_adapter(emb)
        if self.normalize_embeddings:
            out = out / out.norm(dim=1, keepdim=True)
        return out