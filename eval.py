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

    if cfg.style != 'identity':
        assert cfg.unsup_emb not in sup_encs
        assert cfg.unsup_emb in translator.in_adapters
        assert cfg.unsup_emb in translator.out_adapters

        cfg.num_params = sum(x.numel() for x in translator.parameters())
        print("Number of parameters:", cfg.num_params)

    dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    dset = dset_dict["train"]
    valset = dset_dict["test"]

    assert hasattr(cfg, 'num_points') or hasattr(cfg, 'unsup_points')
    dset = dset.shuffle(seed=cfg.train_dataset_seed)
    if hasattr(cfg, 'num_points'):
        assert cfg.num_points > 0 and cfg.num_points <= len(dset) // 2
        supset = dset.select(range(cfg.num_points))
        unsupset = dset.select(range(cfg.num_points, cfg.num_points + cfg.val_size))
    elif hasattr(cfg, 'unsup_points'):
        unsupset = dset.select(range(min(cfg.unsup_points, cfg.val_size)))
        supset = dset.select(range(min(cfg.unsup_points, len(dset)), len(dset) - len(unsupset)))

    num_workers = get_num_proc()
    evalset = MultiencoderTokenizedDataset(
        dataset=supset if hasattr(cfg, 'flip') and cfg.flip else unsupset,
        encoders={ **unsup_enc, **sup_encs },
        n_embs_per_batch=2,
        batch_size=cfg.val_bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )
    evalloader = DataLoader(
        evalset,
        batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    evalloader = accelerator.prepare(evalloader)

    if cfg.style != 'identity':
        assert hasattr(cfg, 'load_dir')
        print(f"Loading models from {argv[1]}...")
        translator.load_state_dict(torch.load(f'{argv[1]}/model.pt', map_location='cpu'), strict=False)

    translator = accelerator.prepare(translator)
    # inverters = get_inverters(["gtr"], accelerator.device)
    inverters = None

    with torch.no_grad():
        translator.eval()
        val_res = {}
        recons, trans, heatmap_dict, text_recons, text_trans, _ =\
            eval_loop_(
                cfg,
                translator,
                {**sup_encs, **unsup_enc},
                evalloader,
                inverters=inverters,
                device=accelerator.device
            )
        val_res['recons'] = {}
        for flag, res in recons.items():
            for k, v in res.items():
                if k == 'cos':
                    val_res['recons'][f"rec_{flag}_{k}"] = v

        val_res['trans'] = {}
        for target_flag, d in trans.items():
            for flag, res in d.items():
                for k, v in res.items():
                    if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                        continue
                    val_res['trans'][f"{flag}_{target_flag}_{k}"] = v

        val_res['heatmap'] = {}
        if len(heatmap_dict) > 0:
            for k,v in heatmap_dict.items():
                if v.__class__.__name__ == 'Figure':
                    continue
                else:
                    val_res['heatmap'][f"{k} (avg. {cfg.top_k_batches} batches)"] = v
        
        val_res['text_recons'] = {}
        if len(text_recons) > 0:
            for flag, res in text_recons.items():
                for k,v in res.items():
                    val_res['text_recons'][f"text_{k}"] = v

        val_res['text_trans'] = {}
        if len(text_trans) > 0:
            for target_flag, d in text_trans.items():
                for flag, res in d.items():
                    for k, v in res.items():
                        if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                            continue
                        val_res['text_trans'][f"{flag}_{target_flag}_{k}"] = v

    if cfg.style == 'identity':
        fnm = f'results/baseline_{cfg.dataset.replace("/", "_")}_{cfg.unsup_emb}_{cfg.sup_emb}.json'
    elif hasattr(cfg, 'flip') and cfg.flip:
        fnm = f'results/{cfg.dataset.replace("/", "_")}_{cfg.sup_emb}_{cfg.unsup_emb}_ood.json'
    else:
        fnm = f'results/{cfg.dataset.replace("/", "_")}_{cfg.unsup_emb}_{cfg.sup_emb}.json'
    with open(fnm, 'w') as f:
        # human readable
        f.write(json.dumps(val_res, indent=4))


if __name__ == "__main__":
    main()