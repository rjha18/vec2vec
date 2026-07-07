"""Geodesic monotonicity probe (fleet 4).

B1 established a 28-sigma gap between L(R_random) and L(R_true), and a
smooth bowl out to 1.5 rad. But descent from >=1.7 rad stalls near its
init. Two incompatible pictures: (a) a global monotone funnel with
transverse traps that gradient flow falls into, or (b) genuine long-range
barriers on every path from random to truth. This decides it: walk the
GEODESIC in O(d) from each of several random rotations to R_true and
record full-cloud (subsampled) energy along the way.

If L(t) is monotone (or nearly) along geodesics -> the funnel is real and
value-based stochastic search (annealing / basin-hopping with many cheap
seeds) is well-posed; descent fails for reasons a Metropolis rule can
survive. If every path has a mid-path hump comparable to the 28-sigma gap
-> long-range search is genuinely blocked and the funnel picture from B1
needs revision (the bowl measured at <=1.5 rad does not extend).

Geodesic: M = Qa^T Qb is orthogonal; its matrix log S (real skew) comes
from the complex eigendecomposition; Q(t) = Qa expm(tS).
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes, logm


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def cross_mean_dist(x, y, bs=2048):
    tot, cnt = 0.0, 0
    for i in range(0, len(x), bs):
        dch = torch.cdist(x[i:i + bs], y)
        tot += float(dch.sum())
        cnt += dch.numel()
    return tot / cnt


def mean_rank(pred, target, bs=2048):
    ranks = []
    for i in range(0, len(pred), bs):
        s = pred[i:i + bs] @ target.T
        d = s.gather(1, torch.arange(i, i + len(s), device=s.device).unsqueeze(1))
        ranks.append((s >= d).sum(1).float())
    return torch.cat(ranks).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True)
    p.add_argument("--emb_b_train", default="")
    p.add_argument("--n_sub", type=int, default=10000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--n_paths", type=int, default=6)
    p.add_argument("--n_ts", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gcpu = torch.Generator().manual_seed(args.seed)
    ea = normalize(torch.load(args.emb_a, map_location="cpu", weights_only=True))
    eb = normalize(torch.load(args.emb_b, map_location="cpu", weights_only=True))
    d = ea.shape[1]

    perm = torch.randperm(len(ea), generator=gcpu)
    ie = perm[:args.n_eval]
    rest = perm[args.n_eval:]
    n = len(rest) // 2
    ia, ib = rest[:n], rest[n:2 * n]
    Xev, Yev = ea[ie].to(device), eb[ie].to(device)
    X = ea[ia][:args.n_sub].to(device)
    if args.emb_b_train:
        Y = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True))[:args.n_sub].to(device)
    else:
        Y = eb[ib][:args.n_sub].to(device)

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qtrue = torch.from_numpy(Wsup.T).float().to(device)

    d_xx = cross_mean_dist(X, X)
    d_yy = cross_mean_dist(Y, Y)

    def L(Q):
        return 2 * cross_mean_dist(X @ Q.T, Y) - d_xx - d_yy

    L_true = L(Qtrue)
    paths = []
    for k in range(args.n_paths):
        G = torch.randn(d, d, generator=torch.Generator().manual_seed(args.seed + 500 + k))
        Qr, Rr = torch.linalg.qr(G)
        Qr = (Qr * torch.sign(torch.diagonal(Rr))).to(device)
        # match determinant component so the geodesic exists in the same O(d) component
        if torch.det(Qr).sign() != torch.det(Qtrue).sign():
            Qr[0] = -Qr[0]
        M = (Qr.T @ Qtrue).cpu().double().numpy()
        S = torch.from_numpy(logm(M).real).float().to(device)
        row = []
        for t in torch.linspace(0, 1, args.n_ts):
            Qt = Qr @ torch.matrix_exp(float(t) * S)
            lv = L(Qt)
            r = mean_rank(normalize(Xev @ Qt.T), Yev)
            row.append({"t": round(float(t), 3), "L": lv, "rank": round(r, 1)})
        ls = [x["L"] for x in row]
        # monotonicity diagnostics: largest uphill move relative to total drop
        max_hump = max(max(ls[i + 1:i + 2][0] - ls[i] for i in range(len(ls) - 1)), 0.0)
        drop = ls[0] - ls[-1]
        paths.append({"path": k, "L_start": ls[0], "L_end": ls[-1],
                      "max_uphill_step": max_hump, "total_drop": drop, "points": row})
        print(f"path {k}: L {ls[0]:.4f} -> {ls[-1]:.4f}, max uphill step {max_hump:.5f}", flush=True)
        print("  L(t):", [round(x, 4) for x in ls], flush=True)

    result = {"emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
              "n_sub": args.n_sub, "L_true": L_true, "paths": paths}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
