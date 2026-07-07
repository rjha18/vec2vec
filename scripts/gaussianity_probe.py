"""UNE-inspired Gaussianity probe + projection-pursuit correspondence test.

Motivated by 'The Universal Normal Embedding' (arXiv:2603.21786, CVPR 2026):
encoder embeddings as noisy linear projections of a shared ~Gaussian latent
(81-92% of random 1-d projections of vision-encoder embeddings pass
normality tests). A whitened Gaussian is maximally rotationally symmetric —
identifiable by NOTHING — so if text embeddings are mostly-Gaussian + thin
non-Gaussian residual, that one assumption predicts our whole phenomenology
(isotropy wall, ICA failure, entropic trap) and gives B3 its cleanest form:
identifiability margin = non-Gaussianity of the shared latent.

Two measurements:

(1) GAUSSIANITY BY SUBSPACE (tests UNE x our direction law): pass rates of
    normality tests (D'Agostino-Pearson, Shapiro-Wilk on 250-pt subsamples,
    their protocol) for random 1-d projections restricted to the top-128
    signal subspace vs the complement vs full space. Prediction: complement
    ~Gaussian (high pass), signal subspace markedly non-Gaussian.

(2) PROJECTION-PURSUIT CORRESPONDENCE (gates the theory-guided matcher):
    find the top-k most-kurtotic unit directions per whitened cloud
    (deflationary pursuit by gradient ascent), then check — using R_true as
    a MEASURING INSTRUMENT — whether pursuit directions correspond across
    clouds (|cos(a_i, R_true^T b_j)| matrix) and whether their 1-d marginal
    shapes would let an unsupervised matcher pair them (quantile-W1 cost
    matrix vs the oracle pairing). If pursuit directions correspond, the
    projection-pursuit matcher is worth building; if not, the non-Gaussian
    residual is encoder-specific and the UNE-shared-skeleton reading fails.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def gaussianity_pass_rates(X, V, n_proj=2000, n_pts=250, seed=0):
    """Random 1-d projections within the row-space of V; normality-test pass rates."""
    g = torch.Generator().manual_seed(seed)
    Xc = X - X.mean(0, keepdim=True)
    dp_pass, sw_pass, kurts = 0, 0, []
    for i in range(n_proj):
        w = torch.randn(V.shape[0], generator=g)
        u = F.normalize(w @ V, dim=0)
        idx = torch.randint(0, len(Xc), (n_pts,), generator=g)
        s = (Xc[idx] @ u).numpy()
        s = (s - s.mean()) / (s.std() + 1e-12)
        if stats.normaltest(s).pvalue > 0.05:
            dp_pass += 1
        if stats.shapiro(s).pvalue > 0.05:
            sw_pass += 1
        kurts.append(float(stats.kurtosis(s)))
    return {"dp_pass": dp_pass / n_proj, "sw_pass": sw_pass / n_proj,
            "mean_abs_kurt": float(np.mean(np.abs(kurts))),
            "p95_abs_kurt": float(np.percentile(np.abs(kurts), 95))}


def pursuit_directions(Z, k=8, iters=300, lr=0.1, seed=0, device="cpu"):
    """Deflationary kurtosis pursuit on a whitened cloud Z (n x d): find k
    orthonormal directions maximizing |excess kurtosis| of the projection."""
    d = Z.shape[1]
    Zd = Z.to(device)
    found = []
    g = torch.Generator().manual_seed(seed)
    for j in range(k):
        best_u, best_k = None, -1.0
        for restart in range(4):
            u = torch.randn(d, generator=g).to(device)
            if found:
                B = torch.stack(found)
                u = u - B.T @ (B @ u)
            u = F.normalize(u, dim=0)
            u = u.clone().requires_grad_(True)
            opt = torch.optim.Adam([u], lr=lr)
            for _ in range(iters):
                un = F.normalize(u, dim=0)
                if found:
                    un = un - B.T @ (B @ un)
                    un = F.normalize(un, dim=0)
                s = Zd @ un
                s = (s - s.mean()) / (s.std() + 1e-8)
                kurt = (s ** 4).mean() - 3.0
                loss = -kurt.abs()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                un = F.normalize(u, dim=0)
                if found:
                    un = un - B.T @ (B @ un)
                    un = F.normalize(un, dim=0)
                s = Zd @ un
                s = (s - s.mean()) / (s.std() + 1e-8)
                kv = float(((s ** 4).mean() - 3.0).abs())
                if kv > best_k:
                    best_k, best_u = kv, un.clone()
        found.append(best_u.detach())
        print(f"  pursuit dir {j}: |kurt| {best_k:.3f}", flush=True)
    return torch.stack(found).cpu()  # (k, d)


def quantile_profile(s, m=512):
    idx = torch.linspace(0, len(s) - 1, m).long()
    return s.sort().values[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True)
    p.add_argument("--emb_b_train", default="")
    p.add_argument("--sig_dim", type=int, default=128)
    p.add_argument("--n_pursuit", type=int, default=8)
    p.add_argument("--n_proj", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gcpu = torch.Generator().manual_seed(args.seed)
    ea = normalize(torch.load(args.emb_a, map_location="cpu", weights_only=True))
    eb = normalize(torch.load(args.emb_b, map_location="cpu", weights_only=True))
    d = ea.shape[1]

    perm = torch.randperm(len(ea), generator=gcpu)
    rest = perm[4096:]
    n = len(rest) // 2
    ia, ib = rest[:n], rest[n:2 * n]
    Xtr = ea[ia]
    Ytr = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True))[:n] \
        if args.emb_b_train else eb[ib]

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    R = torch.from_numpy(Wsup).float()  # x_b ~ x_a @ R

    # subspaces from the A cloud
    Va = torch.linalg.svd(Xtr - Xtr.mean(0, keepdim=True), full_matrices=False).Vh
    V_sig, V_comp = Va[:args.sig_dim], Va[args.sig_dim:]

    print("gaussianity: signal subspace...", flush=True)
    g_sig = gaussianity_pass_rates(Xtr, V_sig, n_proj=args.n_proj, seed=args.seed)
    print(g_sig, flush=True)
    print("gaussianity: complement...", flush=True)
    g_comp = gaussianity_pass_rates(Xtr, V_comp, n_proj=args.n_proj, seed=args.seed)
    print(g_comp, flush=True)
    print("gaussianity: full space...", flush=True)
    g_full = gaussianity_pass_rates(Xtr, torch.eye(d), n_proj=args.n_proj, seed=args.seed)
    print(g_full, flush=True)

    # --- projection pursuit correspondence ---
    def whiten(Xc, k=256):
        mu = Xc.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Xc - mu, full_matrices=False)
        V = Vh[:k]
        s = S[:k] / (len(Xc) - 1) ** 0.5
        return (Xc - mu) @ V.T / s, V, s, mu

    Za, Va_w, sa, mua = whiten(Xtr)
    Zb, Vb_w, sb, mub = whiten(Ytr)
    print("pursuit on A...", flush=True)
    Ua = pursuit_directions(Za, k=args.n_pursuit, seed=args.seed, device=device)
    print("pursuit on B...", flush=True)
    Ub = pursuit_directions(Zb, k=args.n_pursuit, seed=args.seed + 50, device=device)

    # map pursuit directions back to ambient space
    Aamb = F.normalize(Ua / sa @ Va_w, dim=1)   # directions in A ambient (k, d)
    Bamb = F.normalize(Ub / sb @ Vb_w, dim=1)
    # oracle correspondence: |cos(a_i @ R, b_j)|
    C = (Aamb @ R @ Bamb.T).abs()
    oracle_best = C.max(1).values
    # unsupervised matchability: quantile-W1 cost between 1-d marginals
    qa = torch.stack([quantile_profile(Za @ u) for u in Ua])
    qb = torch.stack([quantile_profile(Zb @ u) for u in Ub])
    qb_neg = -qb.flip(dims=[1])
    cost = torch.minimum(torch.cdist(qa, qb, p=1), torch.cdist(qa, qb_neg, p=1)) / qa.shape[1]
    ri, ci = linear_sum_assignment(cost.numpy())
    match_oracle_cos = [round(float(C[i, j]), 3) for i, j in zip(ri, ci)]

    result = {"emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
              "sig_dim": args.sig_dim, "seed": args.seed,
              "gaussianity_signal": g_sig, "gaussianity_complement": g_comp,
              "gaussianity_full": g_full,
              "pursuit_oracle_best_cos": [round(float(x), 3) for x in oracle_best],
              "pursuit_match_cost": [round(float(cost[i, j]), 4) for i, j in zip(ri, ci)],
              "pursuit_matched_oracle_cos": match_oracle_cos}
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("emb")}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
