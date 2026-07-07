"""Energy-distance descent over O(d), v2: big-batch + graduated non-convexity.

Two hypotheses from the v3 record, both about WHY energy descent stalls from
1.5 rad (392 -> ~250 plateau) despite (a) B1's static bowl showing truth beats
random rotations by a large margin out to 1.5 rad and (b) energy's optimum
sitting AT truth even on the conjunction cell:

  H-noise: at bs=2048 the minibatch gradient noise swamps the tiny relational
    signal in the rank-critical ~128-d subspace. Fix = effective batch 32k+
    via gradient accumulation. If H-noise, big-batch descends well below the
    250 plateau at matched step count.

  H-landscape: off-radial local minima / flat plateaus. Fix = graduated
    non-convexity: minimize energy between NOISE-CONVOLVED clouds (add
    N(0, sigma^2 I) to both sides), annealing sigma high -> low. Smoothing
    preserves the minimizer (marginals coincide at R_true per A0) while
    washing out fine structure that creates local traps; each sigma level
    warm-starts the next. This is the assignment-free, non-adversarial
    analogue of the scorepot/GAN coarse-to-fine mechanism.

Baseline for comparison: hb_conj_* / ebasin_conj_* (bs 2048, no noise,
lr 1e-3, 20k steps). Perturbation seeding identical (seed+7) so init ranks
match. Loss semantics identical to utils.train_utils.energy_distance_loss
(row-normalize inside the loss) but implemented inline so this script is
self-contained.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def energy_distance(x, y):
    # matches utils.train_utils.energy_distance_loss: normalize rows, then
    # E-statistic 2 E|x-y| - E|x-x'| - E|y-y'|
    x = normalize(x)
    y = normalize(y)
    d_xy = torch.cdist(x, y, p=2).mean()
    d_xx = torch.cdist(x, x, p=2).mean()
    d_yy = torch.cdist(y, y, p=2).mean()
    return 2.0 * d_xy - d_xx - d_yy


def unbalanced_sinkhorn(x, y, eps=0.1, rho=1.0, n_iters=50):
    """Debiased unbalanced entropic OT (KL-relaxed marginals, Chizat scaling).

    Balanced Sinkhorn drifts on the conjunction cell because its plan MUST
    transport all mass, coupling non-corresponding support. KL relaxation
    (rho) lets unmatched mass be created/destroyed instead. Potentials are
    computed without grad; the loss differentiates through the cost matrix
    with the plan fixed (envelope gradient).
    """
    x, y = normalize(x), normalize(y)

    def uot(u, v):
        C = torch.cdist(u, v).pow(2)
        n_, m_ = C.shape
        la = -torch.log(torch.tensor(float(n_), device=C.device))
        lb = -torch.log(torch.tensor(float(m_), device=C.device))
        lam = rho / (rho + eps)
        with torch.no_grad():
            f = torch.zeros(n_, device=C.device)
            g = torch.zeros(m_, device=C.device)
            for _ in range(n_iters):
                f = -lam * eps * torch.logsumexp((g[None, :] - C) / eps + lb, dim=1)
                g = -lam * eps * torch.logsumexp((f[:, None] - C) / eps + la, dim=0)
            log_pi = (f[:, None] + g[None, :] - C) / eps + la + lb
            pi = log_pi.exp()
        cost = (pi * C).sum()
        with torch.no_grad():
            pa, pb = pi.sum(1), pi.sum(0)
            a, b = torch.full_like(pa, 1.0 / n_), torch.full_like(pb, 1.0 / m_)
            kl_a = (pa * (pa / a).clamp_min(1e-30).log() - pa + a).sum()
            kl_b = (pb * (pb / b).clamp_min(1e-30).log() - pb + b).sum()
        return cost + rho * (kl_a + kl_b)

    return uot(x, y) - 0.5 * uot(x, x) - 0.5 * uot(y, y)


def mean_rank(pred, target, bs=2048):
    ranks = []
    for i in range(0, len(pred), bs):
        s = pred[i:i + bs] @ target.T
        d = s.gather(1, torch.arange(i, i + len(s), device=s.device).unsqueeze(1))
        ranks.append((s >= d).sum(1).float())
    return torch.cat(ranks).mean().item()


def random_rotation_near(d, angle, gen):
    P = torch.randn(d, d, generator=gen)
    skew = P - P.T
    if angle > 0:
        skew = skew / skew.norm() * angle * (d ** 0.5)
    else:
        skew = skew * 0.0
    return torch.matrix_exp(skew)


class OrthMap(nn.Module):
    def __init__(self, d, base, gen):
        super().__init__()
        self.register_buffer("base", base)
        self.P = nn.Parameter(0.001 * torch.randn(d, d, generator=gen))

    def matrix(self):
        skew = self.P - self.P.T
        return torch.matrix_exp(skew) @ self.base

    def forward(self, x):
        return x @ self.matrix().T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True, help="PAIRED with --emb_a (R_true fit + eval)")
    p.add_argument("--emb_b_train", default="", help="conjunction: unpaired cross-corpus target cloud")
    p.add_argument("--objective", choices=["energy", "usinkhorn"], default="energy")
    p.add_argument("--uot_eps", type=float, default=0.1)
    p.add_argument("--uot_rho", type=float, default=1.0)
    p.add_argument("--init", choices=["perturbed_truth", "random"], default="perturbed_truth")
    p.add_argument("--perturb_angle", type=float, default=0.0)
    p.add_argument("--n_train", type=int, default=40000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=2048)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation: effective batch = bs*accum")
    p.add_argument("--sigma_start", type=float, default=0.0, help="GNC: initial noise std added to BOTH clouds")
    p.add_argument("--sigma_end", type=float, default=0.0)
    p.add_argument("--sigma_frac", type=float, default=0.85, help="fraction of steps over which sigma anneals; rest at sigma_end")
    p.add_argument("--icp_after", type=int, default=0,
                   help="if >0: after descent, restart from the best-CSLS checkpoint "
                        "(unsupervised selection) and run this many ICP iterations")
    p.add_argument("--oracle_select", action="store_true",
                   help="MEASURING INSTRUMENT ONLY: select the ICP-restart checkpoint by "
                        "held-out rank instead of CSLS. Decomposes a chain failure into "
                        "selection-failure vs basin-failure; never part of a method.")
    p.add_argument("--csls_sub", type=int, default=8000, help="subsample for the CSLS criterion")
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
    ntr_y = len(Ytr)

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qsup = torch.from_numpy(Wsup.T).float()
    rank_sup = mean_rank(normalize(Xev @ Qsup.T.to(device)), Yev)

    if args.init == "perturbed_truth":
        pert = random_rotation_near(d, args.perturb_angle, torch.Generator().manual_seed(args.seed + 7))
        base = (pert @ Qsup).to(device)
    else:
        # random orthogonal init (moonshot): QR of a gaussian matrix
        G = torch.randn(d, d, generator=torch.Generator().manual_seed(args.seed + 7))
        Qr, R = torch.linalg.qr(G)
        base = (Qr * torch.sign(torch.diagonal(R))).to(device)
    rank_init = mean_rank(normalize(Xev @ base.T), Yev)
    print(f"init ({args.init} pert {args.perturb_angle}): rank {rank_init:.1f}, sup {rank_sup:.2f}", flush=True)

    model = OrthMap(d, base, torch.Generator().manual_seed(args.seed)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def sigma_at(step):
        if args.sigma_start <= 0:
            return 0.0
        t = min(step / max(args.sigma_frac * args.steps, 1), 1.0)
        lo = max(args.sigma_end, 1e-4)
        return float(args.sigma_start * (lo / args.sigma_start) ** t) if args.sigma_end > 0 \
            else float(args.sigma_start * (1.0 - t))

    # unsupervised checkpoint criterion (validated in v2 as a within-family
    # argmax selector): mean CSLS similarity of mutual-NN matches on the
    # unpaired train clouds. The trajectory dip past truth is invisible to
    # the energy value itself (descent keeps lowering it while rank rises),
    # so checkpoints must be selected by an independent unsupervised signal.
    csub = min(args.csls_sub, n, ntr_y)
    Xc, Yc = Xtr[:csub], Ytr[:csub]

    def csls_criterion(Q):
        xq = normalize(Xc @ Q.T)
        s = xq @ Yc.T
        r_x = s.topk(10, dim=1).values.mean(1)
        r_y = s.T.topk(10, dim=1).values.mean(1)
        c = 2 * s - r_x[:, None] - r_y[None, :]
        nn_xy = c.max(1).indices
        nn_yx = c.max(0).indices
        mutual = nn_yx[nn_xy] == torch.arange(len(xq), device=s.device)
        if mutual.sum() == 0:
            return -1e9, 0
        return float(c.max(1).values[mutual].mean()), int(mutual.sum())

    best_crit, best_Q, best_step = -1e9, base.clone(), 0
    best_rank, best_rank_Q, best_rank_step = rank_init, base.clone(), 0
    traj = [(0, round(rank_init, 1), sigma_at(0))]
    for step in range(1, args.steps + 1):
        sig = sigma_at(step)
        opt.zero_grad()
        for _ in range(args.accum):
            xi = Xtr[torch.randint(0, n, (args.bs,), device=device)]
            yi = Ytr[torch.randint(0, ntr_y, (args.bs,), device=device)]
            xm = model(xi)
            if sig > 0:
                xm = xm + sig * torch.randn_like(xm)
                yi = yi + sig * torch.randn_like(yi)
            if args.objective == "usinkhorn":
                loss = unbalanced_sinkhorn(xm, yi, eps=args.uot_eps, rho=args.uot_rho) / args.accum
            else:
                loss = energy_distance(xm, yi) / args.accum
            loss.backward()
        opt.step()
        if step % 250 == 0 or step == args.steps:
            with torch.no_grad():
                Qnow = model.matrix()
                r = mean_rank(normalize(Xev @ Qnow.T), Yev)
                crit, n_mutual = csls_criterion(Qnow)
                if crit > best_crit:
                    best_crit, best_Q, best_step = crit, Qnow.clone(), step
                if r < best_rank:
                    best_rank, best_rank_Q, best_rank_step = r, Qnow.clone(), step
            traj.append((step, round(r, 1), round(sig, 4), round(crit, 4), n_mutual))
            print(f"[energy bs{args.bs}x{args.accum} sig{sig:.3f}] step {step}: rank {r:.1f} "
                  f"csls {crit:.4f} mutual {n_mutual}", flush=True)

    with torch.no_grad():
        Q = model.matrix()
        r_final = mean_rank(normalize(Xev @ Q.T), Yev)
        # rotation angles = complex eigenvalue phases of Q^T Qsup (the SVD
        # formula in objective_descent.py is a null metric on orthogonal M)
        ev = torch.linalg.eigvals((Q.T.cpu() @ Qsup).double())
        drift_deg = float(torch.rad2deg(ev.angle().abs()).mean())

    icp_result = {}
    if args.icp_after > 0:
        # chain: restart from the best-unsupervised-criterion checkpoint and
        # run CSLS mutual-NN ICP (the E1-validated finisher)
        if args.oracle_select:
            Q, sel_step = best_rank_Q, best_rank_step
        else:
            Q, sel_step = best_Q, best_step
        rank_best = mean_rank(normalize(Xev @ Q.T), Yev)
        print(f"chain: selected ({'ORACLE' if args.oracle_select else 'csls'}) step {sel_step}, rank {rank_best:.1f}", flush=True)
        sub = min(n, ntr_y, 20000)
        Xs, Ys = Xtr[:sub], Ytr[:sub]
        icp_traj = []
        for it in range(args.icp_after):
            with torch.no_grad():
                xq = normalize(Xs @ Q.T)
                s = xq @ Ys.T
                r_x = s.topk(10, dim=1).values.mean(1)
                r_y = s.T.topk(10, dim=1).values.mean(1)
                c = 2 * s - r_x[:, None] - r_y[None, :]
                nn_xy = c.max(1).indices
                nn_yx = c.max(0).indices
                keep = nn_yx[nn_xy] == torch.arange(len(xq), device=device)
                pa = torch.arange(len(xq), device=device)[keep]
                pb = nn_xy[keep]
                if len(pa) < 50:
                    break
                W, _ = orthogonal_procrustes(Xs[pa].cpu().numpy(), Ys[pb].cpu().numpy())
                Q = torch.from_numpy(W.T).float().to(device)
                r = mean_rank(normalize(Xev @ Q.T), Yev)
            icp_traj.append((it, round(r, 1), int(len(pa))))
            if it % 5 == 0 or it == args.icp_after - 1:
                print(f"icp {it}: {len(pa)} pairs, rank {r:.1f}", flush=True)
        r_chain = mean_rank(normalize(Xev @ Q.T), Yev)
        icp_result = {"best_csls": best_crit, "best_csls_step": best_step,
                      "oracle_select": args.oracle_select, "selected_step": sel_step,
                      "rank_at_best_csls": rank_best, "rank_after_chain_icp": r_chain,
                      "icp_trajectory": icp_traj}

    result = {
        "emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
        "objective": args.objective, "uot_rho": args.uot_rho,
        "init": args.init, "perturb_angle": args.perturb_angle, "seed": args.seed,
        "n_train": n, "bs": args.bs, "accum": args.accum, "steps": args.steps, "lr": args.lr,
        "sigma_start": args.sigma_start, "sigma_end": args.sigma_end,
        "rank_supervised": rank_sup, "rank_init": rank_init, "rank_final": r_final,
        "drift_deg_from_truth": drift_deg, "trajectory": traj,
        "verdict": "STABLE" if r_final < 5 * max(rank_sup, 1) else ("PARTIAL" if r_final < 100 else "DRIFTS"),
        **icp_result,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory"}))


if __name__ == "__main__":
    main()
