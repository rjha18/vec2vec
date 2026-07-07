"""How destructive are false-positive tethers? (Procrustes contamination sweep)

Sets the PRECISION BAR any tether-identification method must clear. Start
from K oracle-good tethers (gte-NQ -> best gtr-FineWeb proxy) and replace a
fraction p with random (wrong) target matches, then fit Procrustes (+ICP)
and measure held-out rank vs p. Two robustness regimes matter:

- Raw Procrustes: a least-squares orthogonal fit — every wrong pair pulls
  the rotation. Expected to degrade smoothly/steeply with p (breakdown
  point ~ ?).
- Robust variants: (a) trimmed Procrustes (fit, drop worst-residual
  fraction, refit), (b) an ICP pass after the contaminated fit (mutual-NN
  re-selection may wash out injected noise). Tests whether the DOWNSTREAM
  machinery forgives false positives the raw fit does not.

Reading: if rank stays low up to p~0.5, identification only needs ~50%
precision (easy — many weak signals clear it); if it breaks by p~0.1, we
need a high-precision selector and the search is hard. Combined with the
oracle-ceiling K-curve this gives the full (precision x recall) budget.
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


def procrustes_Q(Xsrc, Ytgt, device):
    W, _ = orthogonal_procrustes(Xsrc.cpu().numpy(), Ytgt.cpu().numpy())
    return torch.from_numpy(W.T).float().to(device)


def trimmed_procrustes(Xsrc, Ytgt, device, keep=0.7, rounds=3):
    Q = procrustes_Q(Xsrc, Ytgt, device)
    idx = torch.arange(len(Xsrc), device=device)
    for _ in range(rounds):
        resid = (normalize(Xsrc[idx] @ Q.T) - Ytgt[idx]).pow(2).sum(1)
        m = max(50, int(len(idx) * keep))
        idx = idx[resid.topk(m, largest=False).indices]
        Q = procrustes_Q(Xsrc[idx], Ytgt[idx], device)
    return Q


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


def icp_refine(Q, Xs, Ys, Xev, Yev, iters=40):
    for _ in range(iters):
        with torch.no_grad():
            pa, pb = csls_mutual(normalize(Xs @ Q.T), Ys)
            if len(pa) < 50:
                break
            Q = procrustes_Q(Xs[pa], Ys[pb], Xs.device)
    return mean_rank(normalize(Xev @ Q.T), Yev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True, help="gtr-NQ oracle (paired)")
    p.add_argument("--emb_b_train", required=True, help="gtr-FineWeb target cloud")
    p.add_argument("--n_train", type=int, default=60000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--K", type=int, default=1000, help="tether budget")
    p.add_argument("--fps", default="0.0,0.1,0.2,0.3,0.5,0.7,0.9")
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
    ia = rest[:n]
    Xev, Yev = ea[ie].to(device), eb[ie].to(device)
    Xtr = ea[ia].to(device)
    OB = eb[ia].to(device)
    FW = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True)).to(device)

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    rank_sup = mean_rank(normalize(Xev @ torch.from_numpy(Wsup).float().to(device)), Yev)

    # oracle best proxies, take the top-K by proxy cosine
    prox_idx, prox_cos = [], []
    for i in range(0, n, 2048):
        s = OB[i:i + 2048] @ FW.T
        v, j = s.max(dim=1)
        prox_idx.append(j); prox_cos.append(v)
    prox_idx = torch.cat(prox_idx); prox_cos = torch.cat(prox_cos)
    order = prox_cos.argsort(descending=True)[:args.K]
    Xg = Xtr[order]
    Yg = FW[prox_idx[order]]

    gfp = torch.Generator().manual_seed(args.seed + 3)
    sub = min(n, 20000)
    Xs, Ys = Xtr[:sub], FW[:sub]
    rows = []
    for fp in [float(x) for x in args.fps.split(",")]:
        n_bad = int(args.K * fp)
        Yc = Yg.clone()
        if n_bad > 0:
            bad_rows = torch.randperm(args.K, generator=gfp)[:n_bad]
            bad_tgt = torch.randint(0, len(FW), (n_bad,), generator=gfp)
            Yc[bad_rows] = FW[bad_tgt]
        Qr = procrustes_Q(Xg, Yc, device)
        rank_raw = mean_rank(normalize(Xev @ Qr.T), Yev)
        Qt = trimmed_procrustes(Xg, Yc, device)
        rank_trim = mean_rank(normalize(Xev @ Qt.T), Yev)
        rank_icp = icp_refine(Qr, Xs, Ys, Xev, Yev)
        row = {"fp": fp, "rank_raw": round(rank_raw, 1),
               "rank_trimmed": round(rank_trim, 1), "rank_raw_then_icp": round(rank_icp, 1)}
        rows.append(row); print(row, flush=True)

    result = {"emb_a": args.emb_a, "emb_b_train": args.emb_b_train, "K": args.K,
              "seed": args.seed, "rank_supervised": rank_sup, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
