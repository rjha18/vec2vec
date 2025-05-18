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

from datasets import load_dataset, load_from_disk


def compute_plan_and_match(A, B, method='hungarian', sinkhorn_reg=1e-1):
    n, dA = A.shape
    _, dB = B.shape

    P = None

    if method == 'gromov-wasserstein':
        # build intra‐space cost matrices
        C1 = torch.cdist(A, A, p=2).cpu().numpy()
        C2 = torch.cdist(B, B, p=2).cpu().numpy()
        a = np.ones(n) / n
        b = np.ones(n) / n
        # square_loss is the standard choice
        P = ot.gromov.gromov_wasserstein(C1, C2, a, b, loss_fun='square_loss')
        row_ind = np.arange(n)
        col_ind = P.argmax(axis=1)

    else:
        # classic OT on pairwise cross‐distances
        M = torch.cdist(A, B, p=2).cpu().numpy()
        a = np.ones(n) / n
        b = np.ones(n) / n

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
            raise ValueError(f"Unknown method: {method}")

    # accuracy
    pred = np.empty(n, dtype=int)
    pred[row_ind] = col_ind
    acc = (pred == np.arange(n)).mean()

    # if we have a coupling P, compute rank stats
    if P is not None:
        ranks = []
        for i in range(n):
            order = np.argsort(-P[i])
            rank = np.where(order == i)[0][0] + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        return acc, ranks.mean(), ranks.var()

    return acc, -1, 0


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


    if cfg.dataset == 'tweets':
        dset_name = 'cardiffnlp/tweet_topic_multilingual'
        unsupset = load_dataset(dset_name, 'en', num_proc=8)['test']
    elif cfg.dataset == 'mimic_templates':
        dset_name = 'data/mimic_templates'
        dset = load_from_disk(dset_name)['unsupervised'].shuffle(seed=cfg.val_dataset_seed)
        unsupset = dset.select(range(cfg.val_size))
    else:
        dset = load_streaming_embeddings(cfg.dataset)
        dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
        dset = dset_dict["train"]
        valset = dset_dict["test"]
        assert hasattr(cfg, 'num_points')
        dset = dset.shuffle(seed=cfg.train_dataset_seed)
        if hasattr(cfg, 'num_points'):
            assert cfg.num_points > 0 and cfg.num_points <= len(dset) // 2
            # supset = dset.select(range(cfg.num_points))
            unsupset = dset.select(range(cfg.num_points, cfg.num_points + cfg.val_size))

    sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)}

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {
        cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)
    }


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

# after you prepare `evalloader` and before you enter the loop:
    methods = None
    acc_res = {}
    rank_res = {}
    rank_var_res = {}

    with torch.no_grad():
        for _, batch in enumerate(evalloader):
            ins = process_batch(batch, encoders, cfg.normalize_embeddings, accelerator.device)

            # on first batch, decide which methods to run
            if methods is None:
                dA = ins[cfg.sup_emb].shape[1]
                dB = ins[cfg.unsup_emb].shape[1]
                if dA == dB:
                    methods = [
                        'hungarian',
                        'emd',
                        'emd-hungarian',
                        'sinkhorn',
                        'sinkhorn-hungarian',
                        'gromov-wasserstein'
                    ]
                else:
                    methods = ['gromov-wasserstein']

                # initialize our result lists
                acc_res       = {m: [] for m in methods}
                rank_res      = {m: [] for m in methods}
                rank_var_res  = {f"{m}_var": [] for m in methods}

            # now run all requested methods
            for method in methods:
                acc, rank, rank_var = compute_plan_and_match(
                    ins[cfg.sup_emb],
                    ins[cfg.unsup_emb],
                    method=method,
                    sinkhorn_reg=cfg.sinkhorn_reg if hasattr(cfg, 'sinkhorn_reg') else 1e-1
                )
                acc_res[method].append(acc)
                rank_res[method].append(rank)
                rank_var_res[f"{method}_var"].append(rank_var)

    # aggregate
    for m in methods:
        acc_res[m]       = float(np.mean(acc_res[m]))
        rank_res[m]      = float(np.mean(rank_res[m]))
        rank_var_res[f"{m}_var"] = float(np.mean(rank_var_res[f"{m}_var"]))
    rank_se_res = {
        f"{m}_se": np.sqrt(rank_var_res[f"{m}_var"]) / np.sqrt(len(evalloader) * cfg.val_bs)
        for m in methods
    }

    results = {
        'acc':      {m: acc_res[m] for m in methods},
        'rank':     {m: rank_res[m] for m in methods},
        **{f"{m}_var": rank_var_res[f"{m}_var"] for m in methods},
        **rank_se_res
    }

    # write out only the computed methods
    fnm = f'ot_results/{cfg.dataset}_{cfg.sup_emb}_{cfg.unsup_emb}.json'
    os.makedirs(os.path.dirname(fnm), exist_ok=True)
    with open(fnm, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Ran methods: {methods}")
    print(f"Results saved to {fnm}")

if __name__ == "__main__":
    main()