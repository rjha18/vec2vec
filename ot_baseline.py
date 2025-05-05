import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import torch
import ot
from scipy.optimize import linear_sum_assignment

# from eval import eval_model
from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.dist import get_rank
from utils.model_utils import load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings, process_batch


def compute_plan_and_match(A, B, method='hungarian', sinkhorn_reg=1e-1):
    M = torch.cdist(A, B, p=2).cpu().numpy()
    n = A.shape[0]
    a = np.ones(n) / n
    b = np.ones(n) / n

    P = None
    if method == 'hungarian':
        row_ind, col_ind = linear_sum_assignment(M)
    elif method == 'emd':
        P = ot.emd(a, b, M)
        row_ind = np.arange(n)
        col_ind = P.argmax(axis=1)
    elif method == 'emd-hungarian':
        P = ot.emd(a, b, M)
        row_ind, col_ind = linear_sum_assignment(-P)
    elif method == 'sinkhorn':
        P = ot.sinkhorn(a, b, M, reg=sinkhorn_reg)
        row_ind = np.arange(n)
        col_ind = P.argmax(axis=1)
    elif method == 'sinkhorn-hungarian':
        P = ot.sinkhorn(a, b, M, reg=sinkhorn_reg)
        row_ind, col_ind = linear_sum_assignment(-P)
    else:
        raise ValueError("Invalid matching method")
    
    pred = np.empty(n, dtype=int)
    pred[row_ind] = col_ind
    acc = (pred == np.arange(n)).mean()

    if P is not None:
        ranks = []
        for i in range(n):
            order = np.argsort(-P[i])
            rank = np.where(order == i)[0][0] + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        return acc, ranks.mean()
    return acc, -1


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'configs/{argv[1]}.toml')['general']
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

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {
        cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)
    }

    assert cfg.unsup_emb not in sup_encs

    dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    dset = dset_dict["train"]
    valset = dset_dict["test"]

    assert hasattr(cfg, 'num_points')
    dset = dset.shuffle(seed=cfg.train_dataset_seed)
    if hasattr(cfg, 'num_points'):
        assert cfg.num_points > 0 and cfg.num_points <= len(dset) // 2
        # supset = dset.select(range(cfg.num_points))
        unsupset = dset.select(range(cfg.num_points, cfg.num_points + cfg.val_size))


    num_workers = get_num_proc()
    evalset = MultiencoderTokenizedDataset(
        dataset=valset if hasattr(cfg, 'use_ood') and cfg.use_ood else unsupset,
        encoders={ **unsup_enc, **sup_encs },
        n_embs_per_batch=2,
        batch_size=cfg.val_bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )
    evalloader = DataLoader(
        evalset,
        batch_size=cfg.val_bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    evalloader = accelerator.prepare(evalloader)
    encoders = {**sup_encs, **unsup_enc}

    METHODS = ['hungarian', 'emd', 'emd-hungarian', 'sinkhorn', 'sinkhorn-hungarian']
    acc_res = {method: [] for method in METHODS}
    rank_res = {method: [] for method in METHODS}
    
    with torch.no_grad():
        for _, batch in enumerate(evalloader):
            ins = process_batch(batch, encoders, cfg.normalize_embeddings, accelerator.device)
            for method in METHODS:
                acc, rank = compute_plan_and_match(ins[cfg.sup_emb], ins[cfg.unsup_emb], method=method)
                acc_res[method].append(acc)
                rank_res[method].append(rank)
    
    for method in METHODS:
        acc_res[method] = np.mean(acc_res[method])
        rank_res[method] = np.mean(rank_res[method])

    fnm = f'ot_results/{cfg.dataset}_{cfg.sup_emb}_{cfg.unsup_emb}.json'
    os.makedirs(os.path.dirname(fnm), exist_ok=True)
    with open(fnm, 'w') as f:
        # human readable json
        json.dump({
            'acc': acc_res,
            'rank': rank_res
        }, f, indent=4)
    print(f"Results saved to {fnm}")

if __name__ == "__main__":
    main()