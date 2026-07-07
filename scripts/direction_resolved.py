"""Direction-resolved error geometry (fleet 3, exp A).

Fleet 2's sharpest finding: rank is NOT a basin coordinate. An energy-drifted
rotation at rank 134 is LESS ICP-recoverable (-> 71) than a random
perturbation of truth at rank 392 (-> 41.6). Hypothesis: what matters is how
the rotation error is distributed between the ~128-d retrieval-critical
(signal) subspace and its complement — distribution-descent concentrates
error in the signal directions; random perturbations spread it isotropically.

This script builds a panel of error rotations with known provenance:
  a. random perturbations of R_true at several angles,
  b. snapshots along a big-batch energy descent from a 1.5-rad init
     (the dip path),
  c. SWD-drift from truth (the classic pathology, strongest contrast),
and for each reports: retrieval rank, the fraction of eval-point displacement
inside the signal subspace (top-k PCA of the target cloud), and the ICP
outcome (recoverability label).

Second half: gamma-spectrum lite at R_true — full-cloud energy curvature and
minibatch gradient SNR along signal-plane vs complement-plane vs random skew
directions. This is the empirical margin gamma of the identifiability
statement (B3), direction-resolved.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
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


def energy_distance(x, y):
    x, y = normalize(x), normalize(y)
    return (2.0 * torch.cdist(x, y).mean() - torch.cdist(x, x).mean()
            - torch.cdist(y, y).mean())


def random_rotation_near(d, angle, gen):
    P = torch.randn(d, d, generator=gen)
    skew = P - P.T
    skew = skew / skew.norm() * angle * (d ** 0.5) if angle > 0 else skew * 0.0
    return torch.matrix_exp(skew)


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


def icp(Q, Xs, Ys, Xev, Yev, iters=25):
    for _ in range(iters):
        with torch.no_grad():
            xq = normalize(Xs @ Q.T)
            pa, pb = csls_mutual(xq, Ys)
            if len(pa) < 50:
                break
            W, _ = orthogonal_procrustes(Xs[pa].cpu().numpy(), Ys[pb].cpu().numpy())
            Q = torch.from_numpy(W.T).float().to(Xs.device)
    return mean_rank(normalize(Xev @ Q.T), Yev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True)
    p.add_argument("--emb_b_train", default="")
    p.add_argument("--sig_dim", type=int, default=128)
    p.add_argument("--n_train", type=int, default=40000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--curv_sub", type=int, default=8000)
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
        Ytr = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True))[:n].to(device)
    else:
        Ytr = eb[ib].to(device)

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qsup = torch.from_numpy(Wsup.T).float().to(device)
    rank_sup = mean_rank(normalize(Xev @ Qsup.T), Yev)

    # signal subspace: top-k PCA of the paired target cloud
    Yc = eb[ib] - eb[ib].mean(0, keepdim=True)
    Vb = torch.linalg.svd(Yc, full_matrices=False).Vh  # (d, d)
    V_sig = Vb[:args.sig_dim].to(device)               # (k, d)
    V_low = Vb[-args.sig_dim:].to(device)              # bottom-k (complement probe)

    Ysup = normalize(Xev @ Qsup.T)

    def profile(Q, label, provenance):
        Yq = normalize(Xev @ Q.T)
        r = mean_rank(Yq, Yev)
        D = Yq - Ysup
        tot = D.pow(2).sum()
        frac_sig = float((D @ V_sig.T).pow(2).sum() / tot) if tot > 0 else 0.0
        r_icp = icp(Q.clone(), Xtr[:20000], Ytr[:20000], Xev, Yev)
        row = {"label": label, "provenance": provenance, "rank": round(r, 1),
               "frac_disp_signal": round(frac_sig, 4),
               "sig_frac_expected_random": round(args.sig_dim / d, 4),
               "rank_after_icp": round(r_icp, 1)}
        print(row, flush=True)
        return row

    panel = []
    # (a) random perturbations
    for ang in [0.5, 1.0, 1.3, 1.5, 1.7, 2.0]:
        for s in [0, 1]:
            pert = random_rotation_near(d, ang, torch.Generator().manual_seed(args.seed + 7 + s)).to(device)
            panel.append(profile(pert @ Qsup, f"rand_{ang}_s{s}", "random_perturbation"))

    # (b) energy-descent snapshots from a 1.5-rad init (big batch)
    pert = random_rotation_near(d, 1.5, torch.Generator().manual_seed(args.seed + 7)).to(device)
    base = pert @ Qsup

    class OrthMap(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("base", base)
            self.P = nn.Parameter(0.001 * torch.randn(d, d, generator=torch.Generator().manual_seed(args.seed)))

        def matrix(self):
            return torch.matrix_exp(self.P - self.P.T) @ self.base

    model = OrthMap().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    snap_at = {500, 1000, 2000, 4000, 6000}
    for step in range(1, 6001):
        opt.zero_grad()
        for _ in range(16):
            xi = Xtr[torch.randint(0, n, (2048,), device=device)]
            yi = Ytr[torch.randint(0, len(Ytr), (2048,), device=device)]
            (energy_distance(xi @ model.matrix().T, yi) / 16).backward()
        opt.step()
        if step in snap_at:
            with torch.no_grad():
                panel.append(profile(model.matrix().clone(), f"energy_step{step}", "energy_descent_from_1.5"))

    # (c) SWD drift from truth (classic pathology)
    def swd(x, y, n_proj=512, gen=None):
        theta = normalize(torch.randn(n_proj, d, generator=gen, device=x.device))
        xp = (x @ theta.T).sort(dim=0).values
        yp = (y @ theta.T).sort(dim=0).values
        return (xp - yp).pow(2).mean()

    model2 = OrthMap().to(device)
    model2.base.copy_(Qsup)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    ggpu = torch.Generator(device=device).manual_seed(args.seed + 1)
    for step in range(1, 4001):
        xi = Xtr[torch.randint(0, n, (2048,), device=device)]
        yi = Ytr[torch.randint(0, len(Ytr), (2048,), device=device)]
        loss = swd(normalize(xi @ model2.matrix().T), yi, gen=ggpu)
        opt2.zero_grad(); loss.backward(); opt2.step()
        if step in {2000, 4000}:
            with torch.no_grad():
                panel.append(profile(model2.matrix().clone(), f"swd_step{step}", "swd_drift_from_truth"))

    # --- gamma-spectrum lite: curvature + gradient SNR by direction class ---
    Xc_, Yc_ = Xtr[:args.curv_sub], Ytr[:args.curv_sub]

    def L_full(Q):
        return float(energy_distance(Xc_ @ Q.T, Yc_))

    def plane_skew(u, v):
        S = torch.outer(u, v) - torch.outer(v, u)
        return S / S.norm()

    gdir = torch.Generator().manual_seed(args.seed + 99)
    dirs = []
    for i in range(8):
        u, v = V_sig[2 * i], V_sig[2 * i + 1]
        dirs.append(("signal", plane_skew(u, v)))
    for i in range(8):
        u, v = V_low[2 * i], V_low[2 * i + 1]
        dirs.append(("complement", plane_skew(u, v)))
    for i in range(8):
        P_ = torch.randn(d, d, generator=gdir).to(device)
        S = P_ - P_.T
        dirs.append(("random", S / S.norm()))

    L0 = L_full(Qsup)
    spectrum = []
    eps = 0.05
    for cls, S in dirs:
        S = S.to(device)
        Qp = torch.matrix_exp(eps * S * (d ** 0.5)) @ Qsup
        Qm = torch.matrix_exp(-eps * S * (d ** 0.5)) @ Qsup
        curv = (L_full(Qp) + L_full(Qm) - 2 * L0) / (eps ** 2)
        # minibatch directional-gradient noise at bs 2048
        gs = []
        h = 0.02
        Qh_p = torch.matrix_exp(h * S * (d ** 0.5)) @ Qsup
        Qh_m = torch.matrix_exp(-h * S * (d ** 0.5)) @ Qsup
        for _ in range(16):
            xi = Xtr[torch.randint(0, n, (2048,), device=device)]
            yi = Ytr[torch.randint(0, len(Ytr), (2048,), device=device)]
            gs.append((float(energy_distance(xi @ Qh_p.T, yi))
                       - float(energy_distance(xi @ Qh_m.T, yi))) / (2 * h))
        gs = torch.tensor(gs)
        spectrum.append({"class": cls, "curvature": curv,
                         "grad_mean": float(gs.mean()), "grad_std": float(gs.std()),
                         "snr": float(gs.mean().abs() / (gs.std() + 1e-12))})
        print(spectrum[-1], flush=True)

    result = {"emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
              "sig_dim": args.sig_dim, "seed": args.seed, "rank_supervised": rank_sup,
              "panel": panel, "gamma_spectrum": spectrum, "L0": L0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
