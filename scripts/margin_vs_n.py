"""Margin-vs-N scaling law (fleet 3, exp C).

The identifiability margin gamma (how strongly the target marginal prefers
R_true over random rotations under energy distance) determines whether the
conjunction cell is solvable-at-scale: if the normalized margin GROWS with
sample size N, the problem is a sample-complexity problem (solvable with
bigger clouds); if it saturates small, that is the quantified barrier.

For each N: margin(N) = [mean_r L(R_rand) - L(R_true)] / std_r L(R_rand),
with L = full-batch energy distance between R#A_N and B_N. d_xx / d_yy terms
are rotation-invariant so only the cross term is recomputed per rotation.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def cross_mean_dist(x, y, bs=2048):
    tot, cnt = 0.0, 0
    for i in range(0, len(x), bs):
        dch = torch.cdist(x[i:i + bs], y)
        tot += float(dch.sum())
        cnt += dch.numel()
    return tot / cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True)
    p.add_argument("--emb_b_train", default="")
    p.add_argument("--ns", default="5000,10000,20000,40000")
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--n_rand", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gcpu = torch.Generator().manual_seed(args.seed)
    ea = normalize(torch.load(args.emb_a, map_location="cpu", weights_only=True))
    eb = normalize(torch.load(args.emb_b, map_location="cpu", weights_only=True))
    d = ea.shape[1]

    perm = torch.randperm(len(ea), generator=gcpu)
    rest = perm[args.n_eval:]
    n_max = len(rest) // 2
    ia, ib = rest[:n_max], rest[n_max:2 * n_max]

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qsup = torch.from_numpy(Wsup.T).float().to(device)

    if args.emb_b_train:
        Yfull = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True)).to(device)
    else:
        Yfull = eb[ib].to(device)
    Xfull = ea[ia].to(device)

    rand_qs = []
    for r in range(args.n_rand):
        G = torch.randn(d, d, generator=torch.Generator().manual_seed(args.seed + 1000 + r))
        Qr, R_ = torch.linalg.qr(G)
        rand_qs.append((Qr * torch.sign(torch.diagonal(R_))).to(device))

    rows = []
    for N in [int(x) for x in args.ns.split(",")]:
        N = min(N, len(Xfull), len(Yfull))
        X, Y = Xfull[:N], Yfull[:N]
        # rotation-invariant terms
        d_xx = cross_mean_dist(X, X)
        d_yy = cross_mean_dist(Y, Y)

        def L(Q):
            return 2 * cross_mean_dist(normalize(X @ Q.T), Y) - d_xx - d_yy

        L_true = L(Qsup)
        L_rand = torch.tensor([L(Q) for Q in rand_qs])
        margin = float((L_rand.mean() - L_true) / (L_rand.std() + 1e-12))
        row = {"N": N, "L_true": L_true, "L_rand_mean": float(L_rand.mean()),
               "L_rand_std": float(L_rand.std()), "gap": float(L_rand.mean() - L_true),
               "margin_sigmas": margin}
        rows.append(row)
        print(row, flush=True)

    result = {"emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
              "seed": args.seed, "n_rand": args.n_rand, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
