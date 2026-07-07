"""ICP basin sweep with support for the CONJUNCTION cell (v3 gap-fill).

v2's icpbasin_* measured the ICP refiner's basin (1.5 rad, wide) on the
SAME-DATASET pair only (gte-NQ vs gtr-NQ disjoint halves). The v3 program's
localization claim ("only the coarse matcher is missing; ICP finishes from
1.5 rad") therefore rests on an unmeasured assumption for the conjunction
cell — and v3's own assignment-free principle (SWD/Sinkhorn drift under
support mismatch because they impose correspondences; energy does not)
predicts ICP, the most assignment-based refiner of all (hard mutual-NN
pairs), should degrade there.

This script = orthogonal_align.py's ICP branch + objective_descent.py's
data plumbing (--emb_b_train for an unpaired cross-corpus target cloud),
with identical perturbation seeding (seed+7) so init ranks are directly
comparable to the ebasin_conj_* energy-basin numbers.

Reading: if ICP from 1.0-1.5 rad converges to ~rank 1 on the conjunction
cell, the localization claim survives and the coarse-matcher budget is
1.5 rad. If it stalls/drifts, the true budget is energy's ~0.5 rad (3x
harder) and "the finisher" is also open cross-dataset.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def mean_rank(pred, target, bs=2048):
    ranks = []
    for i in range(0, len(pred), bs):
        s = pred[i:i + bs] @ target.T
        d = s.gather(1, torch.arange(i, i + len(s), device=s.device).unsqueeze(1))
        ranks.append((s >= d).sum(1).float())
    return torch.cat(ranks).mean().item()


def random_rotation_near(d, angle, gen):
    # identical to objective_descent.py so init ranks match ebasin_* runs
    P = torch.randn(d, d, generator=gen)
    skew = P - P.T
    if angle > 0:
        skew = skew / skew.norm() * angle * (d ** 0.5)
    else:
        skew = skew * 0.0
    return torch.matrix_exp(skew)


def csls_mutual(xq, y, k=10, keep_frac=1.0):
    """Mutual-NN tethers under CSLS; optionally keep only the top keep_frac
    by CSLS score (dynamic tether selection: precision over recall)."""
    device = xq.device
    r_x = (xq @ y.T).topk(k, dim=1).values.mean(1)
    r_y = (y @ xq.T).topk(k, dim=1).values.mean(1)
    s = 2 * (xq @ y.T) - r_x[:, None] - r_y[None, :]
    nn_xy = s.max(1).indices
    nn_yx = s.max(0).indices
    keep = nn_yx[nn_xy] == torch.arange(len(xq), device=device)
    pa = torch.arange(len(xq), device=device)[keep]
    pb = nn_xy[keep]
    if keep_frac < 1.0 and len(pa) > 50:
        scores = s[pa, pb]
        m = max(50, int(len(pa) * keep_frac))
        top = scores.topk(m).indices
        pa, pb = pa[top], pb[top]
    return pa, pb


def angle_to(Q, Qref):
    """Mean rotation angle between two orthogonal maps.

    NOTE: the SVD-of-M formula used by objective_descent.py is a null metric —
    M = Q^T Qref is orthogonal, so its singular values are identically 1 and
    arccos reads ~0 deg regardless of drift (the ledger's 0.2-0.7 deg 'drift'
    numbers are float noise). The rotation angles are the complex phases of
    M's eigenvalues.
    """
    M = (Q.T.cpu() @ Qref.cpu()).double()
    ev = torch.linalg.eigvals(M)
    return float(torch.rad2deg(ev.angle().abs()).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True, help="source, PAIRED with --emb_b (R_true fit + eval)")
    p.add_argument("--emb_b", required=True, help="target, PAIRED with --emb_a")
    p.add_argument("--emb_b_train", default="", help="conjunction: unpaired cross-corpus target cloud "
                   "for the ICP matching; empty = --emb_b's disjoint half (same-dataset)")
    p.add_argument("--perturb_angle", type=float, required=True)
    p.add_argument("--n_train", type=int, default=40000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--icp_iters", type=int, default=40)
    p.add_argument("--icp_sub", type=int, default=20000)
    p.add_argument("--csls_k", type=int, default=10)
    p.add_argument("--keep_frac", type=float, default=1.0,
                   help="dynamic tether selection: keep only this top fraction of mutual pairs by CSLS score")
    p.add_argument("--oracle_b", default="",
                   help="MEASURING INSTRUMENT: emb_b-space embeddings of emb_a's texts (e.g. gtr-NQ), "
                        "row-aligned with emb_a, for per-tether precision@10 logging")
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
    n = min(args.n_train, len(rest) // 2)
    ia, ib = rest[:n], rest[n:2 * n]
    Xev, Yev = ea[ie].to(device), eb[ie].to(device)

    Xtr = ea[ia].to(device)
    if args.emb_b_train:
        eb_train = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True))
        Ytr = eb_train[:n].to(device)
    else:
        Ytr = eb[ib].to(device)

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qsup = torch.from_numpy(Wsup.T).float()
    rank_sup = mean_rank(normalize(Xev @ Qsup.T.to(device)), Yev)

    pert = random_rotation_near(d, args.perturb_angle, torch.Generator().manual_seed(args.seed + 7))
    Q = (pert @ Qsup).to(device)
    rank_init = mean_rank(normalize(Xev @ Q.T), Yev)
    print(f"init (pert {args.perturb_angle}): rank {rank_init:.1f}, sup {rank_sup:.2f}", flush=True)

    sub = min(n, len(Ytr), args.icp_sub)
    Xs, Ys = Xtr[:sub], Ytr[:sub]

    # oracle tether precision: matched target j is 'good' if Ys[j] is within
    # the top-10 target-cloud neighbors of the source's TRUE b-space embedding
    oracle_top = None
    if args.oracle_b:
        ob = normalize(torch.load(args.oracle_b, map_location="cpu", weights_only=True))
        ob_sub = ob[ia][:sub].to(device)
        tops = []
        for i in range(0, sub, 2048):
            tops.append((ob_sub[i:i + 2048] @ Ys.T).topk(10, dim=1).indices)
        oracle_top = torch.cat(tops)

    traj = [(-1, round(rank_init, 1), 0, None)]
    for it in range(args.icp_iters):
        with torch.no_grad():
            xq = normalize(Xs @ Q.T)
            pa, pb = csls_mutual(xq, Ys, k=args.csls_k, keep_frac=args.keep_frac)
            if len(pa) < 50:
                print(f"icp {it}: only {len(pa)} pairs, stopping", flush=True)
                break
            prec = None
            if oracle_top is not None:
                prec = round(float((oracle_top[pa] == pb[:, None]).any(1).float().mean()), 4)
            W, _ = orthogonal_procrustes(Xs[pa].cpu().numpy(), Ys[pb].cpu().numpy())
            Q = torch.from_numpy(W.T).float().to(device)
            r = mean_rank(normalize(Xev @ Q.T), Yev)
        traj.append((it, round(r, 1), int(len(pa)), prec))
        print(f"icp {it}: {len(pa)} pairs, rank {r:.1f}, tether precision@10 {prec}", flush=True)

    r_final = mean_rank(normalize(Xev @ Q.T), Yev)
    result = {
        "emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
        "keep_frac": args.keep_frac, "oracle_b": args.oracle_b,
        "perturb_angle": args.perturb_angle, "seed": args.seed, "n_train": n,
        "icp_sub": sub, "rank_supervised": rank_sup, "rank_init": rank_init,
        "rank_final": r_final, "drift_deg_from_truth": angle_to(Q, Qsup),
        "trajectory": traj,
        "verdict": "STABLE" if r_final < 5 * max(rank_sup, 1) else ("PARTIAL" if r_final < 100 else "DRIFTS"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory"}))


if __name__ == "__main__":
    main()
