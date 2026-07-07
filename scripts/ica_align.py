"""ICA / fourth-order-cumulant coarse matcher (first higher-order method).

Every relational coarse matcher tried so far — GW, cov-axes, spectral,
QAP-on-centroid-Grams — is SECOND-order, and the shared "isotropy wall"
diagnosis (near-degenerate covariance => no canonical axes) is a
second-order statement. But ICA theory (Comon) says a rotation acting on a
whitened NON-Gaussian distribution is identifiable from higher cumulants,
and embedding clouds are strongly non-Gaussian (cluster structure).
Literature support: ICA axes of independently trained embedding models
correspond across models/languages (Yamagiwa et al. 2023).

Method (fully unsupervised on the target pair, no requery):
  1. Per cloud: center, top-k PCA, whiten -> z (n x k).
  2. FastICA per cloud (whiten=False) -> near-orthogonal unmixing W (k x k),
     sources s = z @ W.T with approximately independent non-Gaussian axes.
  3. Match A-axes to B-axes by the W1 distance between their 1-d marginal
     quantile profiles (rotation-invariant descriptors), Hungarian assignment,
     sign resolved by trying both orientations.
  4. Compose the whitened-space rotation Wa.T @ (signed permutation) @ Wb,
     map back to B's ambient frame -> affine coarse map.
  5. Extract orthogonal Procrustes approximation Q0 and refine with CSLS
     mutual-NN ICP (same refiner as icp_basin.py).

Cells: gte<->e5 within-lineage (positive control, MUST land in basin),
gte<->gtr same-dataset (the cross-lineage test), conjunction (gte-NQ vs
gtr-FineWeb via --emb_b_train). Supervised R_true is a measuring instrument
(rank ceiling + angle diagnostic) only.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def mean_rank(pred, target, bs=2048):
    ranks = []
    for i in range(0, len(pred), bs):
        s = pred[i:i + bs] @ target.T
        d = s.gather(1, torch.arange(i, i + len(s), device=s.device).unsqueeze(1))
        ranks.append((s >= d).sum(1).float())
    return torch.cat(ranks).mean().item()


def csls_mutual(xq, y, k=10):
    device = xq.device
    r_x = (xq @ y.T).topk(k, dim=1).values.mean(1)
    r_y = (y @ xq.T).topk(k, dim=1).values.mean(1)
    s = 2 * (xq @ y.T) - r_x[:, None] - r_y[None, :]
    nn_xy = s.max(1).indices
    nn_yx = s.max(0).indices
    keep = nn_yx[nn_xy] == torch.arange(len(xq), device=device)
    return torch.arange(len(xq), device=device)[keep], nn_xy[keep]


def whiten_fit(X, k):
    """Return (mu, V (k x d), s (k,)) so that z = (X - mu) @ V.T / s is white."""
    mu = X.mean(0, keepdim=True)
    C = X - mu
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    V = Vh[:k]
    s = S[:k] / (len(X) - 1) ** 0.5
    return mu, V, s


def ica_axes(z, seed, max_iter):
    """FastICA on pre-whitened z; returns orthogonalized unmixing W (k x k)."""
    k = z.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=k, whiten=False, max_iter=max_iter, tol=1e-5,
                      random_state=seed)
        ica.fit(z.numpy().astype(np.float64))
    W = torch.from_numpy(ica.components_).float()
    # project to the nearest orthogonal matrix (FastICA drifts slightly)
    U, _, Vh = torch.linalg.svd(W)
    return U @ Vh


def quantile_profile(src, m=1024):
    """(n x k) sources -> (k x m) sorted quantile profiles per axis."""
    n, k = src.shape
    idx = torch.linspace(0, n - 1, m).long()
    return src.sort(dim=0).values[idx].T.contiguous()  # (k, m)


def match_axes(qa, qb):
    """Hungarian match of axis marginals with sign resolution.

    cost[i,j] = min over sign of mean |quantiles(a_i) - quantiles(+-b_j)|.
    Note sorted(-x) = -reverse(sorted(x)). Returns the signed permutation M,
    the mean matched cost, and the per-match costs (for confidence gating).
    """
    qb_neg = -qb.flip(dims=[1])
    c_pos = torch.cdist(qa, qb, p=1) / qa.shape[1]
    c_neg = torch.cdist(qa, qb_neg, p=1) / qa.shape[1]
    cost = torch.minimum(c_pos, c_neg)
    sign = torch.where(c_pos <= c_neg, 1.0, -1.0)
    ri, ci = linear_sum_assignment(cost.numpy())
    k = qa.shape[0]
    M = torch.zeros(k, k)
    row_cost = torch.zeros(k)
    for i, j in zip(ri, ci):
        M[i, j] = sign[i, j]
        row_cost[i] = cost[i, j]
    matched_cost = float(cost[ri, ci].mean())
    return M, matched_cost, row_cost


def polar(A):
    U, _, Vh = torch.linalg.svd(A)
    return U @ Vh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True, help="PAIRED with --emb_a (R_true fit + eval)")
    p.add_argument("--emb_b_train", default="", help="conjunction: unpaired cross-corpus target cloud")
    p.add_argument("--pca_dim", type=int, default=128)
    p.add_argument("--n_train", type=int, default=45000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--ica_max_iter", type=int, default=2000)
    p.add_argument("--ica_restarts", type=int, default=3)
    p.add_argument("--n_quantiles", type=int, default=1024)
    p.add_argument("--icp_iters", type=int, default=40)
    p.add_argument("--icp_sub", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gcpu = torch.Generator().manual_seed(args.seed)
    ea = normalize(torch.load(args.emb_a, map_location="cpu", weights_only=True))
    eb = normalize(torch.load(args.emb_b, map_location="cpu", weights_only=True))
    d = ea.shape[1]
    k = args.pca_dim

    perm = torch.randperm(len(ea), generator=gcpu)
    ie = perm[:args.n_eval]
    rest = perm[args.n_eval:]
    n = min(args.n_train, len(rest) // 2)
    ia, ib = rest[:n], rest[n:2 * n]
    Xev, Yev = ea[ie].to(device), eb[ie].to(device)

    Xtr = ea[ia]
    if args.emb_b_train:
        eb_train = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True))
        Ytr = eb_train[:min(n, len(eb_train))]
    else:
        Ytr = eb[ib]

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qsup = torch.from_numpy(Wsup.T).float()
    rank_sup = mean_rank(normalize(Xev @ Qsup.T.to(device)), Yev)
    print(f"supervised ceiling: rank {rank_sup:.2f}", flush=True)

    # --- per-cloud whitening + ICA ---
    mu_a, Va, sa = whiten_fit(Xtr, k)
    mu_b, Vb, sb = whiten_fit(Ytr, k)
    za = (Xtr - mu_a) @ Va.T / sa
    zb = (Ytr - mu_b) @ Vb.T / sb

    # FastICA is sample-unstable (local optima): fit several restarts per cloud
    # and keep the (A-run, B-run) pair whose matched-marginal cost is lowest.
    Was, Wbs = [], []
    for r in range(args.ica_restarts):
        print(f"fitting FastICA restart {r}...", flush=True)
        Was.append(ica_axes(za, args.seed + r, args.ica_max_iter))
        Wbs.append(ica_axes(zb, args.seed + 100 + r, args.ica_max_iter))
    best = None
    for Wa_c in Was:
        qa = quantile_profile(za @ Wa_c.T, args.n_quantiles)
        for Wb_c in Wbs:
            qb = quantile_profile(zb @ Wb_c.T, args.n_quantiles)
            M_c, cost_c, row_c = match_axes(qa, qb)
            if best is None or cost_c < best[0]:
                best = (cost_c, M_c, row_c, Wa_c, Wb_c)
    matched_cost, M, row_cost, Wa, Wb = best
    src_a, src_b = za @ Wa.T, zb @ Wb.T

    # non-Gaussianity diagnostic: excess kurtosis per axis
    def kurt(s):
        s = (s - s.mean(0)) / s.std(0)
        return (s ** 4).mean(0) - 3.0
    ka, kb = kurt(src_a), kurt(src_b)
    print(f"axis match: mean matched W1 {matched_cost:.4f}; "
          f"|kurt|>0.3 axes A {(ka.abs() > 0.3).sum().item()}/{k}, B {(kb.abs() > 0.3).sum().item()}/{k}", flush=True)

    # --- compose coarse map: A ambient -> B ambient (affine through whitened frames) ---
    # Confidence gating: fit the whitened rotation ONLY on well-matched axes
    # (below-median marginal cost); noisy tail matches otherwise corrupt the fit.
    thresh = row_cost.median()
    conf = row_cost <= thresh
    M_sel = M * conf[:, None]
    n_conf = int(conf.sum())
    R_white = polar(Wa.T @ M_sel @ Wb)  # za -> zb, orthogonal
    print(f"confident axes used for rotation fit: {n_conf}/{k}", flush=True)

    def ica_map(X):
        z = (X.cpu() - mu_a) @ Va.T / sa
        zb_pred = z @ R_white
        return (zb_pred * sb) @ Vb + mu_b

    rank_ica = mean_rank(normalize(ica_map(Xev)).to(device), Yev)
    print(f"ICA affine coarse map: rank {rank_ica:.1f}", flush=True)

    # --- orthogonal extraction + ICP refine in ambient space ---
    mapped_tr = normalize(ica_map(Xtr))
    W0, _ = orthogonal_procrustes(Xtr.numpy(), mapped_tr.numpy())
    Q = torch.from_numpy(W0.T).float().to(device)
    rank_q0 = mean_rank(normalize(Xev @ Q.T), Yev)
    # rotation angles = complex eigenvalue phases (SVD formula is a null metric)
    ev = torch.linalg.eigvals((Q.T.cpu() @ Qsup).double())
    angle_q0 = float(torch.rad2deg(ev.angle().abs()).mean())
    print(f"orthogonalized coarse map Q0: rank {rank_q0:.1f}, angle to truth {angle_q0:.1f} deg", flush=True)

    sub = min(len(Xtr), len(Ytr), args.icp_sub)
    Xs, Ys = Xtr[:sub].to(device), Ytr[:sub].to(device)
    traj = [(-1, round(rank_q0, 1), 0)]
    for it in range(args.icp_iters):
        with torch.no_grad():
            xq = normalize(Xs @ Q.T)
            pa, pb = csls_mutual(xq, Ys)
            if len(pa) < 50:
                print(f"icp {it}: only {len(pa)} pairs, stopping", flush=True)
                break
            W, _ = orthogonal_procrustes(Xs[pa].cpu().numpy(), Ys[pb].cpu().numpy())
            Q = torch.from_numpy(W.T).float().to(device)
            r = mean_rank(normalize(Xev @ Q.T), Yev)
        traj.append((it, round(r, 1), int(len(pa))))
        print(f"icp {it}: {len(pa)} pairs, rank {r:.1f}", flush=True)
    r_final = mean_rank(normalize(Xev @ Q.T), Yev)

    result = {
        "emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
        "pca_dim": k, "seed": args.seed, "n_train": n,
        "rank_supervised": rank_sup, "rank_ica_affine": rank_ica,
        "rank_q0": rank_q0, "angle_q0_deg": angle_q0,
        "rank_final_after_icp": r_final, "matched_axis_w1": matched_cost,
        "confident_axes": n_conf, "ica_restarts": args.ica_restarts,
        "kurtotic_axes_a": int((ka.abs() > 0.3).sum()), "kurtotic_axes_b": int((kb.abs() > 0.3).sum()),
        "icp_trajectory": traj,
        "verdict": "SOLVED" if r_final < 5 * max(rank_sup, 1) else ("PARTIAL" if r_final < 100 else "FAIL"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps({kk: v for kk, v in result.items() if kk != "icp_trajectory"}))


if __name__ == "__main__":
    main()
