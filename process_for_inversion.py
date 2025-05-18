import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import pandas as pd
import torch

# from eval import eval_model
from utils.collate import MultiEncoderClassificationDataset, TokenizedCollator
from utils.dist import get_rank
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import process_batch

from datasets import load_dataset
import re
import pickle



def preprocess_email(text: str, max_tokens=32) -> str:
    # 1) Drop forwarded blocks (e.g., Original or Forwarded messages)
    text = re.split(r'-{2,}\s*(Original Message|Forwarded message)\s*-{2,}', text)[0]

    # 2) Remove lines starting with To:, Cc:, Bcc:, From:, or Date:
    text = re.sub(
        r'(?m)^(?:\s*(?:[Tt]o|[Cc][Cc]|[Bb]cc|[Ss]ubject|[Ff]rom|[Dd]ate)\s*:).*(\n|$)',
        '',
        text
    )

    # 2.5) Remove the first instance of the word "Body: "
    text = re.sub(
        r'(?i)Body:\s*',
        '',
        text,
        count=1
    )

    # 3) Remove all email addresses
    text = re.sub(
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        '',
        text
    )

    # 4) Preserve newlines as placeholder tokens before splitting
    placeholder = '<NL>'
    text = text.replace('\n', f' {placeholder} ')

    # 5) Tokenize on whitespace & truncate to first 32 tokens
    tokens = text.split()
    truncated = tokens[:max_tokens]

    # 6) Reassemble, restoring placeholders back to newlines
    result = ' '.join(truncated)
    result = result.replace(placeholder, '\n')
    return result.strip()



def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'{argv[1]}/config.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**cfg, **unknown_cfg})

    if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        cfg.mixed_precision = 'fp16'
        print("Note: bf16 is not available on this hardware!")

    # set seeds
    random.seed(cfg.seed + get_rank())
    torch.manual_seed(cfg.seed + get_rank())
    np.random.seed(cfg.seed + get_rank())
    torch.cuda.manual_seed(cfg.seed + get_rank())

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None
    )
    # https://github.com/huggingface/transformers/issues/26548
    accelerator.dataloader_config.dispatch_batches = False

    sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision != 'no' else None)}
    encoder_dims = {cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])}
    translator = load_n_translator(cfg, encoder_dims)

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {
        cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision != 'no' else None)
    }
    unsup_dim = {
        cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb])
    }
    translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])

    assert cfg.unsup_emb not in sup_encs
    assert cfg.unsup_emb in translator.in_adapters
    assert cfg.unsup_emb in translator.out_adapters

    cfg.num_params = sum(x.numel() for x in translator.parameters())
    print("Number of parameters:", cfg.num_params)

    ### Tweets
    # dset = load_dataset('cardiffnlp/tweet_topic_multilingual', 'en', num_proc=8)['test']
    dset = load_dataset('rishi-jha/filtered_enron', split='train', num_proc=8).shuffle(seed=cfg.val_dataset_seed).select(range(128))
    num_workers = get_num_proc()

    dset = MultiEncoderClassificationDataset(
        dataset=dset,
        encoders={ **unsup_enc, **sup_encs },
        n_embs_per_batch=2,
        batch_size=cfg.val_bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
        return_texts=True,
    )

    valloader = DataLoader(
        dset,
        batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=TokenizedCollator(),
        drop_last=False,
    )
    valloader = accelerator.prepare(valloader)

    assert hasattr(cfg, 'load_dir')
    print(f"Loading models from {argv[1]}...")
    translator.load_state_dict(torch.load(f'{argv[1]}/model.pt', map_location='cpu'), strict=False)

    translator = accelerator.prepare(translator)

    with torch.no_grad():
        translator.eval()
        for i, batch in enumerate(valloader):
            ins = process_batch(batch, {**sup_encs, **unsup_enc}, cfg.normalize_embeddings, accelerator.device)

            _, translations = translator(ins, include_reps=False)
            embs = {}
            embs[f'{cfg.unsup_emb}->{cfg.sup_emb}'] = translations[cfg.sup_emb][cfg.unsup_emb]
            embs[f'{cfg.sup_emb}->{cfg.unsup_emb}'] = translations[cfg.unsup_emb][cfg.sup_emb]
            embs[f'text'] = [preprocess_email(x) for x in batch[f'text']]

            with open('sample.txt', 'w') as fout:
                for rec in embs[f'text']:
                    fout.write(rec + '\n\n')


            with open(f'pkls_new_body/{cfg.unsup_emb}_{cfg.sup_emb}.pkl', 'wb') as f:
                pickle.dump(embs, f)
            exit()



if __name__ == "__main__":
    main()

