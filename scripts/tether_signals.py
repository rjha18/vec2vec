"""What unsupervised signal identifies GOOD tethers? (signal hunt)

CSLS margin is proven blind (tether-law result: top-30% by CSLS gave zero
precision gain). This ranks candidate mutual-NN tethers by several
UNSUPERVISED signals and scores each against the oracle (gtr-NQ) via the
precision@k of its top-ranked tethers. A signal with high top-decile
precision is a usable selector; the search then reduces to thresholding it.

Candidate signals (all computable from the two clouds + current Q only):
  - csls_margin: baseline (expected ~flat, the null).
  - gap12: cosine gap between the 1st and 2nd target neighbor (peakedness /
    non-ambiguity of the match).
  - local_density: inverse of mean distance to own-cloud kNN on BOTH sides
    (hubs are unreliable; isolated points match more distinctively).
  - cycle: does mapping A->B->A return near the source? (cycle-consistency
    at the point level).
  - nn_overlap: Jaccard of the source point's mapped kNN set vs the target
    match's kNN set (local-neighborhood agreement -- ARH-style topology,
    the signal that was support-robust in v2's knn_graph probe).
  - reciprocal_rank: rank of source in target's neighbor list x vice versa.

Uses a MODERATE init (perturbed truth) so tethers are informative-but-noisy
(the regime a real search sits in), and reports per-signal precision@{10%,
25%} + Spearman(signal, oracle_correct). Oracle = gtr-NQ, instrument only.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def random_rotation_near(d, angle, gen):
    P = torch.randn(d, d, generator=gen)
    skew = P - P.T
    skew = skew / skew.norm() * angle * (d ** 0.5) if angle > 0 else skew * 0.0
    return torch.matrix_exp(skew)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True, help="gtr-NQ oracle (paired)")
    p.add_argument("--emb_b_train", required=True, help="gtr-FineWeb target")
    p.add_argument("--perturb_angle", type=float, default=1.0,
                   help="init = truth perturbed this much (informative-noisy tether regime)")
    p.add_argument("--n_train", type=int, default=40000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--sub", type=int, default=15000)
    p.add_argument("--knn", type=int, default=10)
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
    n = min(args.n_train, len(rest) // 2)
    ia = rest[:n]
    Xtr = ea[ia].to(device)
    OB = eb[ia].to(device)
    FW = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True)).to(device)
    sub = min(n, len(FW), args.sub)
    X, OBs, Y = Xtr[:sub], OB[:sub], FW[:sub]

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qsup = torch.from_numpy(Wsup.T).float().to(device)
    pert = random_rotation_near(d, args.perturb_angle, torch.Generator().manual_seed(args.seed + 7)).to(device)
    Q = pert @ Qsup

    xq = normalize(X @ Q.T)
    S = xq @ Y.T                                       # (sub, sub)
    k = args.knn
    # mutual-NN candidate tethers under CSLS
    r_x = S.topk(k, dim=1).values.mean(1)
    r_y = S.T.topk(k, dim=1).values.mean(1)
    C = 2 * S - r_x[:, None] - r_y[None, :]
    nn_xy = C.max(1).indices
    nn_yx = C.max(0).indices
    mutual = nn_yx[nn_xy] == torch.arange(sub, device=device)
    src = torch.arange(sub, device=device)[mutual]
    tgt = nn_xy[mutual]
    m = len(src)
    print(f"{m} mutual-NN candidate tethers at pert {args.perturb_angle}", flush=True)

    # oracle correctness: matched target within top-10 of the TRUE embedding's Y-neighbors
    oracle_top = (OBs[src] @ Y.T).topk(10, dim=1).indices
    correct = (oracle_top == tgt[:, None]).any(1).float()
    base_prec = float(correct.mean())
    print(f"baseline mutual-NN precision@10: {base_prec:.4f}", flush=True)

    # --- signals ---
    sig = {}
    sig["csls_margin"] = C[src, tgt]
    top2 = S[src].topk(2, dim=1).values
    sig["gap12"] = top2[:, 0] - top2[:, 1]
    # local density (own-cloud): mean cos to kNN; lower = more isolated/distinctive
    dens_x = (xq[src] @ xq.T).topk(k + 1, dim=1).values[:, 1:].mean(1)
    dens_y = (Y[tgt] @ Y.T).topk(k + 1, dim=1).values[:, 1:].mean(1)
    sig["neg_density"] = -(dens_x + dens_y)
    # cycle consistency: B->A nearest then compare
    back = normalize(Y[tgt] @ Q)                       # map target back to A-space (Q^T)
    sig["cycle"] = (back * X[src]).sum(1)
    # neighborhood overlap (ARH topology): Jaccard of mapped-A kNN vs target kNN (index-aligned via tgt of neighbors)
    xq_nbr = xq[src].topk(k + 1, dim=1) if False else None
    A_nbr = (xq[src] @ Y.T).topk(k, dim=1).indices     # target-cloud neighbors of mapped source
    B_nbr = (Y[tgt] @ Y.T).topk(k + 1, dim=1).indices[:, 1:]
    ov = torch.zeros(m, device=device)
    for i in range(m):
        ov[i] = len(set(A_nbr[i].tolist()) & set(B_nbr[i].tolist()))
    sig["nn_overlap"] = ov

    rows = {}
    for name, s in sig.items():
        s = s.float()
        order = s.argsort(descending=True)
        p10 = float(correct[order[:max(10, m // 10)]].mean())
        p25 = float(correct[order[:max(10, m // 4)]].mean())
        rho = float(spearmanr(s.cpu().numpy(), correct.cpu().numpy()).statistic)
        rows[name] = {"prec_top10pct": round(p10, 4), "prec_top25pct": round(p25, 4),
                      "spearman_vs_correct": round(rho, 4), "lift_top10pct": round(p10 / max(base_prec, 1e-6), 2)}
        print(name.ljust(14), rows[name], flush=True)

    result = {"emb_a": args.emb_a, "emb_b_train": args.emb_b_train,
              "perturb_angle": args.perturb_angle, "seed": args.seed,
              "n_candidates": m, "baseline_precision": round(base_prec, 4), "signals": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
