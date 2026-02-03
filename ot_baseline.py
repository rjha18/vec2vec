import os
import random
import toml
import json
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import scipy as sp
import torch
import ot
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

# from eval import eval_model
from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.dist import get_rank
from utils.model_utils import load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings, process_batch

from datasets import load_dataset, load_from_disk


def compute_se(var, n_batches, batch_size):
    """Compute standard error given variance and number of samples."""
    return np.sqrt(var) / np.sqrt(n_batches * batch_size)


def compute_one_to_one_accuracy(P, n, mode='hungarian'):
    if mode == 'hungarian':
        row_ind, col_ind = linear_sum_assignment(-P)
    elif mode == 'direct':
        row_ind = np.arange(n)
        col_ind = P.argmax(axis=1)
    pred = np.empty(n, dtype=int)
    pred[row_ind] = col_ind
    return (pred == np.arange(n)).mean()

def compute_rank(P, n):
    return np.array([
        np.where(np.argsort(-P[i]) == i)[0][0] + 1
        for i in range(n)
    ])

def compute_barycentric_similarity(P, B):
    B_np = B.cpu().numpy()
    n = P.shape[0]
    B_prime_np = P.dot(B_np) * n
    B_prime = torch.from_numpy(B_prime_np).to(B.device)
    return torch.nn.functional.cosine_similarity(B, B_prime, dim=1).cpu().numpy()


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

    if 'clip' in cfg.sup_emb or 'clip' in cfg.unsup_emb or 'qwen' in cfg.sup_emb or 'qwen' in cfg.unsup_emb:
        methods = ['gromov-wasserstein']
        dims_mismatch = True
    else:
        methods = ['emd', 'sinkhorn', 'gromov-wasserstein']
        dims_mismatch = False

    # methods = ['gromov-wasserstein']

    acc_res = {m: [] for m in methods + [f'{m}-hungarian' for m in methods]}
    rank_res = {m: [] for m in methods}
    rank_var_res = {m: [] for m in methods}
    cosine_sims_res = {m: [] for m in methods}
    cosine_sims_var_res = {m: [] for m in methods}

    if not dims_mismatch:
        acc_res['hungarian'] = []

    reg = cfg.sinkhorn_reg if hasattr(cfg, 'sinkhorn_reg') else 5e-3

    # after you prepare `evalloader` and before you enter the loop:
    with torch.no_grad():
        for _, batch in enumerate(evalloader):
            ins = process_batch(batch, encoders, cfg.normalize_embeddings, accelerator.device)
            A = ins[cfg.sup_emb].cpu()
            B = ins[cfg.unsup_emb].cpu()
            # C1 = (torch.cdist(A, A) ** 2).cpu().numpy()
            # C2 = (torch.cdist(B, B) ** 2).cpu().numpy()

            C1 = sp.spatial.distance.cdist(A, A, metric='cosine')
            C2 = sp.spatial.distance.cdist(B, B, metric='cosine')
            C1 /= C1.mean()
            C2 /= C2.mean()

            a = np.ones(len(A)) / len(A)
            b = np.ones(len(B)) / len(B)
            n = len(A)
            print("\n\nGW\n\n")
            # gw = ot.gromov.gromov_wasserstein(C1, C2, a, b, loss_fun='square_loss', verbose=True)
            gw = G = ot.gromov.entropic_gromov_wasserstein(C1, C2, a, b,
                                                            loss_fun = 'square_loss',
                                                            max_iter = 10000,
                                                            tol = 1e-9,
                                                            epsilon = 5e-3,
                                                            verbose = True,
                                                            numItermax = 10000)
            print("\n\nGW done\n\n")
            
            Ps = {}

            if dims_mismatch:
                Ps['gromov-wasserstein'] = gw
            else:
                M = torch.cdist(A, B, p=2).cpu().numpy()
                
                Ps['gromov-wasserstein'] = gw
                Ps['emd'] = ot.emd(a, b, M)
                Ps['sinkhorn'] = ot.sinkhorn(a, b, M, reg=5e-2, verbose=True, numItermax=10000)

                acc_res['hungarian'].append(compute_one_to_one_accuracy(-M, n, mode='hungarian'))

            for method, P in Ps.items():
                acc_res[method].append(compute_one_to_one_accuracy(P, n, mode='direct'))
                acc_res[f'{method}-hungarian'].append(compute_one_to_one_accuracy(P, n, mode='hungarian'))

                ranks = compute_rank(P, n)
                rank_res[method].append(ranks.mean())
                rank_var_res[method].append(ranks.var())

                cos_sims = compute_barycentric_similarity(P, B)
                cosine_sims_res[method].append(cos_sims.mean())
                cosine_sims_var_res[method].append(cos_sims.var())

    # for each res dict, take the mean of all leaf nodes (acc_res has extra keys)
    rank_se_res = {}
    cosine_sims_se_res = {}
    for m in methods:
        acc_res[m] = float(np.mean(acc_res[m]))
        acc_res[f'{m}-hungarian'] = float(np.mean(acc_res[f'{m}-hungarian']))
        rank_res[m] = float(np.mean(rank_res[m]))
        rank_var_res[m] = float(np.mean(rank_var_res[m]))
        rank_se_res[m] = compute_se(rank_var_res[m], len(evalloader), cfg.val_bs)
        cosine_sims_res[m] = float(np.mean(cosine_sims_res[m]))
        cosine_sims_var_res[m] = float(np.mean(cosine_sims_var_res[m]))
        cosine_sims_se_res[m] = compute_se(cosine_sims_var_res[m], len(evalloader), cfg.val_bs)
    
    if not dims_mismatch:
        acc_res['hungarian'] = float(np.mean(acc_res['hungarian']))

    # combine into one dict
    results = {
        'acc': acc_res,
        'rank': rank_res,
        'rank_var': rank_var_res,
        'cosine_sim': cosine_sims_res,
        'cosine_sim_var': cosine_sims_var_res,
        'rank_se': rank_se_res,
        'cosine_sim_se': cosine_sims_se_res,
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