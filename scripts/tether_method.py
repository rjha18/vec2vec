"""THE assembled cold-start unsupervised method (real-data test, no oracle).

Composes everything the tether experiments imply into one deployable method
and reports its number from a genuine cold start (no oracle, no
label-derived init):

  for each of R random restarts:
     init Q = random orthogonal (optionally in the signal subspace)
     bootstrap: mutual-NN tethers under CSLS -> TRIMMED (robust) Procrustes
       -> repeat (FP-tolerant fit; the sweep showed raw/ICP tolerate ~50% FP)
     score the converged Q by an UNSUPERVISED criterion (mean CSLS of mutual
       tethers -- the validated solved-run detector)
  select the best restart by that criterion; report its held-out rank.

This is the honest cold-start counterpart to the oracle ceiling. Prior
(from v2 random-restart failure + the entropic trap + signal collapse at
>=1.5 rad): likely FAILS because no restart lands within the basin in
SO(768). But it produces the actual METHOD number and, via the
criterion-vs-rank scatter across restarts, shows whether selection could
work IF a restart ever ignited. Oracle rank logged per restart as an
instrument only (never used for selection).
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


def mutual_tethers(xq, y, k=10):
    s = xq @ y.T
    r_x = s.topk(k, dim=1).values.mean(1)
    r_y = s.T.topk(k, dim=1).values.mean(1)
    c = 2 * s - r_x[:, None] - r_y[None, :]
    nn_xy = c.max(1).indices
    nn_yx = c.max(0).indices
    keep = nn_yx[nn_xy] == torch.arange(len(xq), device=xq.device)
    idx = torch.arange(len(xq), device=xq.device)[keep]
    return idx, nn_xy[keep], c[idx, nn_xy[keep]]


def bootstrap(Q, Xs, Ys, iters=40, trim=0.7):
    """Trimmed-Procrustes tether bootstrap; returns (Q, mean-CSLS criterion)."""
    crit = -1e9
    for _ in range(iters):
        pa, pb, sc = mutual_tethers(normalize(Xs @ Q.T), Ys)
        if len(pa) < 50:
            break
        # trim worst-residual tethers (robust fit)
        Qtmp = procrustes_Q(Xs[pa], Ys[pb], Xs.device)
        resid = (normalize(Xs[pa] @ Qtmp.T) - Ys[pb]).pow(2).sum(1)
        m = max(50, int(len(pa) * trim))
        good = resid.topk(m, largest=False).indices
        Q = procrustes_Q(Xs[pa][good], Ys[pb][good], Xs.device)
        crit = float(sc.mean())
    return Q, crit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True, help="gtr-NQ (paired eval + oracle instrument)")
    p.add_argument("--emb_b_train", default="", help="gtr-FineWeb (conjunction target)")
    p.add_argument("--restarts", type=int, default=64)
    p.add_argument("--sig_dim", type=int, default=0, help="if >0, restrict random init to signal subspace of target")
    p.add_argument("--n_train", type=int, default=40000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--sub", type=int, default=15000)
    p.add_argument("--boot_iters", type=int, default=40)
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
    Ytr = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True))[:n].to(device) \
        if args.emb_b_train else eb[ib].to(device)

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qsup = torch.from_numpy(Wsup.T).float().to(device)
    rank_sup = mean_rank(normalize(Xev @ Qsup.T), Yev)

    Vsig = None
    if args.sig_dim > 0:
        Yc = (Ytr - Ytr.mean(0, keepdim=True)).cpu()
        Vsig = torch.linalg.svd(Yc, full_matrices=False).Vh[:args.sig_dim].to(device)

    sub = min(n, len(Ytr), args.sub)
    Xs, Ys = Xtr[:sub], Ytr[:sub]

    restarts = []
    for r in range(args.restarts):
        G = torch.randn(d, d, generator=torch.Generator().manual_seed(args.seed + 100 + r))
        Q0, _ = torch.linalg.qr(G)
        Q0 = Q0.to(device)
        if torch.det(Q0).sign() != torch.det(Qsup).sign():
            Q0[0] = -Q0[0]
        Q, crit = bootstrap(Q0, Xs, Ys, iters=args.boot_iters)
        rk = mean_rank(normalize(Xev @ Q.T), Yev)   # oracle instrument only
        restarts.append({"restart": r, "criterion": round(crit, 5), "rank": round(rk, 1)})
        if r % 8 == 0:
            print(restarts[-1], flush=True)

    valid = [x for x in restarts if x["criterion"] > -1e8]
    best = max(valid, key=lambda x: x["criterion"]) if valid else None
    best_rank_any = min(restarts, key=lambda x: x["rank"])
    result = {"emb_a": args.emb_a, "emb_b_train": args.emb_b_train,
              "restarts": args.restarts, "sig_dim": args.sig_dim, "seed": args.seed,
              "rank_supervised": rank_sup,
              "selected_by_criterion": best,
              "best_rank_any_restart": best_rank_any,
              "all_restarts": restarts,
              "verdict": "SOLVED" if best and best["rank"] < 5 * max(rank_sup, 1)
                         else ("SELECTION_MISS" if best_rank_any["rank"] < 100 else "FAIL")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "all_restarts"}))


if __name__ == "__main__":
    main()
