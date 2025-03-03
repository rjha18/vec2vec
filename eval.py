import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import torch

# from eval import eval_model
from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.dist import get_rank
from utils.eval_utils import eval_loop_
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings


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

    dset = load_streaming_embeddings(cfg.dataset)

    sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)}
    encoder_dims = {cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])}
    translator = load_n_translator(cfg, encoder_dims)

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {
        cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)
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

    dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    valset = dset_dict["test"]

    num_workers = get_num_proc()
    valset = MultiencoderTokenizedDataset(
        dataset=valset,
        encoders={ **unsup_enc, **sup_encs },
        n_embs_per_batch=2,
        batch_size=cfg.val_bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )
    valloader = DataLoader(
        valset,
        batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    valloader = accelerator.prepare(valloader)

    assert hasattr(cfg, 'load_dir')
    print(f"Loading models from {argv[1]}...")
    translator.load_state_dict(torch.load(f'{argv[1]}/model.pt', map_location='cpu'), strict=False)

    translator = accelerator.prepare(translator)
    inverters = get_inverters(["gtr"], accelerator.device)

    with torch.no_grad():
        translator.eval()
        val_res = {}
        recons, trans, heatmap_dict, text_recons, text_trans =\
            eval_loop_(cfg, translator, {**sup_encs, **unsup_enc}, valloader, inverters=inverters, device=accelerator.device)
        for flag, res in recons.items():
            for k, v in res.items():
                if k == 'cos':
                    val_res[f"val/rec_{flag}_{k}"] = v
        for target_flag, d in trans.items():
            for flag, res in d.items():
                for k, v in res.items():
                    if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                        continue
                    val_res[f"val/{flag}_{target_flag}_{k}"] = v

        if len(heatmap_dict) > 0:
            for k,v in heatmap_dict.items():
                if k in ["heatmap", "heatmap_softmax"]:
                    val_res[f"val/{k}"] = v
                else:
                    val_res[f"val/{k} (avg. {cfg.top_k_batches} batches)"] = v
        
        if len(text_recons) > 0:
            for flag, res in text_recons.items():
                for k,v in res.items():
                    val_res[f"val/text_{k}"] = v

        if len(text_trans) > 0:
            for target_flag, d in text_trans.items():
                for flag, res in d.items():
                    for k, v in res.items():
                        if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                            continue
                        val_res[f"val/{flag}_{target_flag}_{k}"] = v
        
    print("Validation Results:")
    for k, v in val_res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()