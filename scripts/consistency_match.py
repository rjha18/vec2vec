"""Consistency-graph correspondence distillation (fleet 3, exp B).

The bridge sweep says ~200-500 correct pairs solve the conjunction cell, and
E1 says ICP tolerates approximate cross-corpus pairs. So a coarse matcher
need not produce a rotation — only a candidate-pair pool with a few-percent
precision, followed by outlier-robust filtering. v2's precision matcher
failed because it fed mostly-wrong descriptor matches RAW into Procrustes.
The registration literature's standard fix (spectral matching, TEASER-style)
was never tried: correct pairs are mutually CONSISTENT under any isometry
(|d_A(i,i') - d_B(j,j')| ~ 0, rotation-invariant), so build a consistency
graph over candidates and extract the dominant consistent set via its
leading eigenvector — this tolerates ~95% outliers in low-d. The high-d risk
is distance concentration (the same isotropy that kills GW), so the
within-lineage control gates interpretation, per program discipline.

Pipeline: per-point descriptors (sorted own-cloud kNN distance profile)
-> top candidate pairs by descriptor distance (capped per point)
-> consistency graph W[m,n] = exp(-(dA-dB)^2/tau^2) -> power iteration
-> greedy one-to-one top pairs -> Procrustes -> rank -> ICP refine.
"""
import argparse
import json
from pathlib import Path

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


def csls_mutual(xq, y, k=10):
    device = xq.device
    s = xq @ y.T
    r_x = s.topk(k, dim=1).values.mean(1)
    r_y = s.T.topk(k, dim=1).values.mean(1)
    c = 2 * s - r_x[:, None] - r_y[None, :]
    nn_xy = c.max(1).indices
    nn_yx = c.max(0).indices
    keep = nn_yx[nn_xy] == torch.arange(len(xq), device=device)
    return torch.arange(len(xq), device=device)[keep], nn_xy[keep]


def knn_profile(pts, cloud, k):
    """Sorted distances from each pts row to its k nearest neighbors in cloud."""
    profs = []
    for i in range(0, len(pts), 1024):
        dch = torch.cdist(pts[i:i + 1024], cloud)
        profs.append(dch.topk(k + 1, largest=False).values[:, 1:])  # drop self/0
    return torch.cat(profs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True)
    p.add_argument("--emb_b_train", default="")
    p.add_argument("--n_train", type=int, default=40000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--n_pts", type=int, default=6000, help="candidate points per side")
    p.add_argument("--desc_k", type=int, default=50)
    p.add_argument("--cand_per_pt", type=int, default=4)
    p.add_argument("--n_cand_max", type=int, default=12000)
    p.add_argument("--n_keep", type=int, default=500)
    p.add_argument("--power_iters", type=int, default=60)
    p.add_argument("--icp_iters", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gcpu = torch.Generator().manual_seed(args.seed)
    ea = normalize(torch.load(args.emb_a, map_location="cpu", weights_only=True))
    eb = normalize(torch.load(args.emb_b, map_location="cpu", weights_only=True))

    perm = torch.randperm(len(ea), generator=gcpu)
    ie = perm[:args.n_eval]
    rest = perm[args.n_eval:]
    n = min(args.n_train, len(rest) // 2)
    ia, ib = rest[:n], rest[n:2 * n]
    Xev, Yev = ea[ie].to(device), eb[ie].to(device)
    Xtr = ea[ia].to(device)
    if args.emb_b_train:
        Ytr = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True))[:n].to(device)
    else:
        Ytr = eb[ib].to(device)

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qsup = torch.from_numpy(Wsup.T).float().to(device)
    rank_sup = mean_rank(normalize(Xev @ Qsup.T), Yev)
    print(f"sup ceiling {rank_sup:.2f}", flush=True)

    # --- candidate generation by rotation-invariant descriptors ---
    m = min(args.n_pts, len(Xtr), len(Ytr))
    A, B = Xtr[:m], Ytr[:m]
    da = knn_profile(A, Xtr, args.desc_k)
    db = knn_profile(B, Ytr, args.desc_k)
    # z-score descriptor dims to equalize scales across clouds
    da = (da - da.mean(0)) / (da.std(0) + 1e-8)
    db = (db - db.mean(0)) / (db.std(0) + 1e-8)
    Dd = torch.cdist(da, db)                          # (m, m) descriptor distances
    vals, cols = Dd.topk(args.cand_per_pt, dim=1, largest=False)
    cand_i = torch.arange(m, device=device).repeat_interleave(args.cand_per_pt)
    cand_j = cols.reshape(-1)
    cand_v = vals.reshape(-1)
    if len(cand_i) > args.n_cand_max:
        keep = cand_v.topk(args.n_cand_max, largest=False).indices
        cand_i, cand_j = cand_i[keep], cand_j[keep]
    M = len(cand_i)
    print(f"{M} candidate pairs", flush=True)

    # --- consistency graph + power iteration ---
    dA = torch.cdist(A[cand_i], A[cand_i])            # (M, M)
    dB = torch.cdist(B[cand_j], B[cand_j])
    diff = (dA - dB).abs()
    tau = diff.median()
    W = torch.exp(-(diff / (tau + 1e-8)) ** 2)
    W.fill_diagonal_(0)
    v = torch.ones(M, device=device) / M ** 0.5
    for _ in range(args.power_iters):
        v = W @ v
        v = v / v.norm()
    # greedy one-to-one selection by eigenvector score
    order = v.argsort(descending=True)
    used_a, used_b, sel = set(), set(), []
    for idx in order.tolist():
        i, j = int(cand_i[idx]), int(cand_j[idx])
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); sel.append(idx)
        if len(sel) >= args.n_keep:
            break
    sel = torch.tensor(sel, device=device)
    print(f"selected {len(sel)} pairs (eigenvector mass {float(v[sel].sum()/v.sum()):.3f})", flush=True)

    W0, _ = orthogonal_procrustes(A[cand_i[sel]].cpu().numpy(), B[cand_j[sel]].cpu().numpy())
    Q = torch.from_numpy(W0.T).float().to(device)
    rank_q0 = mean_rank(normalize(Xev @ Q.T), Yev)
    print(f"coarse Procrustes on distilled pairs: rank {rank_q0:.1f}", flush=True)

    traj = []
    sub = min(n, len(Ytr), 20000)
    Xs, Ys = Xtr[:sub], Ytr[:sub]
    for it in range(args.icp_iters):
        with torch.no_grad():
            xq = normalize(Xs @ Q.T)
            pa, pb = csls_mutual(xq, Ys)
            if len(pa) < 50:
                break
            Wi, _ = orthogonal_procrustes(Xs[pa].cpu().numpy(), Ys[pb].cpu().numpy())
            Q = torch.from_numpy(Wi.T).float().to(device)
            r = mean_rank(normalize(Xev @ Q.T), Yev)
        traj.append((it, round(r, 1), int(len(pa))))
        if it % 5 == 0:
            print(f"icp {it}: {len(pa)} pairs, rank {r:.1f}", flush=True)
    r_final = mean_rank(normalize(Xev @ Q.T), Yev)

    result = {"emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
              "seed": args.seed, "n_pts": m, "n_candidates": M, "n_selected": int(len(sel)),
              "rank_supervised": rank_sup, "rank_coarse": rank_q0,
              "rank_final_after_icp": r_final, "icp_trajectory": traj,
              "verdict": "SOLVED" if r_final < 5 * max(rank_sup, 1) else ("PARTIAL" if r_final < 100 else "FAIL")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps({k: val for k, val in result.items() if k != "icp_trajectory"}))


if __name__ == "__main__":
    main()
