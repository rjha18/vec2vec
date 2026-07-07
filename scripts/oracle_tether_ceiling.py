"""Oracle tether ceiling: how good can ANY tether-based method get on the
conjunction cell?

For each source point (gte-NQ), the oracle (its TRUE gtr embedding, cached
gtr-NQ — measuring instrument) finds the closest point that actually exists
in the target cloud (gtr-FineWeb): the best possible cross-corpus proxy.
These are the best tethers any selection mechanism could ever produce.
Procrustes on the top-K (by proxy closeness), optional ICP refinement, and
rank vs K / vs proxy-quality threshold gives the CEILING of the tether
mechanism under truly disjoint supports.

Readings:
- ceiling rank ~1-10: the conjunction cell is solvable by tether selection
  alone -> the open problem reduces to a SEARCH problem (find ~500 good
  tethers without the oracle; CSLS is not the signal, per the tether-law
  result, but the information exists).
- ceiling poor: even perfect correspondence mining cannot beat support
  mismatch -> the tether route is capped; quantifies the support-coverage
  barrier directly.
By-product: the proxy-cosine distribution IS the effective support overlap
of the corpus pair (fraction of NQ points with a close FineWeb proxy) —
sharper than marginal-coincidence measures.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True, help="gte-NQ (paired with --emb_b)")
    p.add_argument("--emb_b", required=True, help="gtr-NQ (oracle, paired)")
    p.add_argument("--emb_b_train", required=True, help="gtr-FineWeb (the actual target cloud)")
    p.add_argument("--n_train", type=int, default=60000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--ks", default="200,500,1000,5000,20000,all")
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
    ia = rest[:n]
    Xev, Yev = ea[ie].to(device), eb[ie].to(device)
    Xtr = ea[ia].to(device)
    OB = eb[ia].to(device)  # oracle true b-embeddings of train rows
    FW = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True)).to(device)

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    rank_sup = mean_rank(normalize(Xev @ torch.from_numpy(Wsup).float().to(device)), Yev)

    # oracle best proxies: nearest FW point to each true embedding
    prox_idx, prox_cos = [], []
    for i in range(0, n, 2048):
        s = OB[i:i + 2048] @ FW.T
        v, j = s.max(dim=1)
        prox_idx.append(j)
        prox_cos.append(v)
    prox_idx = torch.cat(prox_idx)
    prox_cos = torch.cat(prox_cos)
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    cos_quantiles = {f"q{int(x*100)}": round(float(torch.quantile(prox_cos, x)), 4) for x in q}
    frac_close = {f"cos>={t}": round(float((prox_cos >= t).float().mean()), 4) for t in [0.7, 0.8, 0.9, 0.95]}
    print("proxy-cos quantiles:", cos_quantiles, flush=True)
    print("effective overlap:", frac_close, flush=True)

    order = prox_cos.argsort(descending=True)
    rows = []
    for kspec in args.ks.split(","):
        K = n if kspec == "all" else int(kspec)
        sel = order[:K]
        W0, _ = orthogonal_procrustes(Xtr[sel].cpu().numpy(), FW[prox_idx[sel]].cpu().numpy())
        Q = torch.from_numpy(W0.T).float().to(device)
        r0 = mean_rank(normalize(Xev @ Q.T), Yev)
        # unsupervised ICP refinement from the oracle-tether rotation
        sub = min(n, 20000)
        Xs, Ys = Xtr[:sub], FW[:sub]
        for _ in range(args.icp_iters):
            with torch.no_grad():
                pa, pb = csls_mutual(normalize(Xs @ Q.T), Ys)
                if len(pa) < 50:
                    break
                Wi, _ = orthogonal_procrustes(Xs[pa].cpu().numpy(), Ys[pb].cpu().numpy())
                Q = torch.from_numpy(Wi.T).float().to(device)
        r_icp = mean_rank(normalize(Xev @ Q.T), Yev)
        row = {"K": K, "min_proxy_cos": round(float(prox_cos[sel].min()), 4),
               "rank_procrustes": round(r0, 1), "rank_after_icp": round(r_icp, 1)}
        rows.append(row)
        print(row, flush=True)

    result = {"emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
              "seed": args.seed, "n_train": n, "n_fw": len(FW),
              "rank_supervised": rank_sup, "proxy_cos_quantiles": cos_quantiles,
              "effective_overlap": frac_close, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
