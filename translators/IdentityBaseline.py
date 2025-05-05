import torch
from translators.AbsNTranslator import AbsNTranslator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IdentityBaseline(AbsNTranslator):
    def __init__(
        self,
        encoder_dims: dict[str, int],
    ):
        super().__init__(encoder_dims, 0, 0)

    def translate_embeddings(
        self, embeddings: torch.Tensor, in_name: str, out_name: str,
    ) -> torch.Tensor:
        return embeddings
    
    def _make_adapters(self):
        pass

    def add_encoders(self, encoder_dims: dict[str, int], overwrite_embs: list[str] = None):
        pass

    def forward(
        self,
        ins: dict[str, torch.Tensor],
        in_set: set[str] = None,
        out_set: set[str] = None,
        include_reps: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        in_set = in_set if in_set is not None else ins.keys()
        out_set = out_set if out_set is not None else ins.keys()

        translations = {}

        for flag in in_set:
            for target_flag in out_set:
                if target_flag != flag:
                    if target_flag not in translations: translations[target_flag] = {}
                    translations[target_flag][flag] = ins[flag]


        return ins, translations
