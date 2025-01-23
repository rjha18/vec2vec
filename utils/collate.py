import random
import torch
import transformers

from utils.tokenization import get_tokenizer_max_length

class MultiEncoderCollator(transformers.DataCollatorWithPadding):
    def __init__(self, encoders, n_embs_per_batch: int = 2, max_length: int = 32):
        self.tokenizers = { enc_name: enc.tokenizer for enc_name, enc in encoders.items() }
        self.n_embs_per_batch = n_embs_per_batch
        self.max_length = max_length
        
    def __call__(self, ex_list):
        tok_names = random.sample(list(self.tokenizers.keys()), k=self.n_embs_per_batch)
        output = {}
        tokenizer_max_lengths = [get_tokenizer_max_length(self.tokenizers[tok_name]) for tok_name in tok_names]
        max_length = min(*tokenizer_max_lengths, self.max_length)

        ex_text = [ex["text"] for ex in ex_list]

        for tok_name in tok_names:
            tt = self.tokenizers[tok_name](
                ex_text,
                truncation="longest_first",
                padding="max_length", 
                max_length=max_length, 
                return_tensors="pt"
            )
            if get_tokenizer_max_length(self.tokenizers[tok_name]) == 77:
                # TODO: Use a less hacky way to figure out that this is a CLIP model :-)
                tt["image_text_info"] = torch.ones(len(ex_list))

            output.update({ f"{tok_name}_{key}": value for key, value in tt.items()})
        return output
