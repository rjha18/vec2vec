"""Funnel-SA: value-based stochastic search assembled from the program's priors.

Priors used (all measured, fleets 1-3):
  1. The energy landscape has a 28-sigma global gap between random rotations
     and truth (B1 + margin_vs_n), i.e. a macro-funnel — IF paths to truth
     are not blocked (geodesic_probe tests this).
  2. Gradient FOLLOWING fails: descent's continuous path concentrates error
     into signal directions (direction poisoning, E5b). A Metropolis rule on
     the energy VALUE makes jumps, accepts occasional uphill moves, and
     never integrates a poisoned path.
  3. Only the ~128-d signal subspace is constrained (1000x curvature ratio),
     so moves are random plane rotations WITHIN span(V_sig-ish planes) —
     collapsing the search from SO(768) to an effective SO(128).
  4. ICP finishes from within ~1.5 rad / rank~400 (E1), and the CSLS
     mutual-NN count is a validated detector of basin entry (v2 selection
     result) — so SA only needs to get CLOSE, then hands off.
  5. Runs are cheap (subsampled full-cloud energy, closed-form rank-2
     rotations) -> many seeds + principled selection is permissible.

Per-proposal cost ~ one 8k x 8k cdist. Rank-2 plane rotation uses the
closed form expm(theta*S) = I + sin(theta) S + (1-cos(theta)) S^2 (S^3=-S),
applied as a rank-2 update — no 768x768 matrix_exp.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def cross_mean_dist(x, y, bs=4096):
    tot, cnt = 0.0, 0
    for i in range(0, len(x), bs):
        dch = torch.cdist(x[i:i + bs], y)
        tot += float(dch.sum())
        cnt += dch.numel()
    return tot / cnt


def mean_rank(pred, target, bs=2048):
    ranks = []
    for i in range(0, len(pred), bs):
        s = pred[i:i + bs] @ target.T
        d = s.gather(1, torch.arange(i, i + len(s), device=s.device).unsqueeze(1))
        ranks.append((s >= d).sum(1).float())
    return torch.cat(ranks).mean().item()


def csls_mutual_count(xq, y, k=10):
    s = xq @ y.T
    r_x = s.topk(k, dim=1).values.mean(1)
    r_y = s.T.topk(k, dim=1).values.mean(1)
    c = 2 * s - r_x[:, None] - r_y[None, :]
    nn_xy = c.max(1).indices
    nn_yx = c.max(0).indices
    return int((nn_yx[nn_xy] == torch.arange(len(xq), device=s.device)).sum())


def csls_mutual(xq, y, k=10):
    s = xq @ y.T
    r_x = s.topk(k, dim=1).values.mean(1)
    r_y = s.T.topk(k, dim=1).values.mean(1)
    c = 2 * s - r_x[:, None] - r_y[None, :]
    nn_xy = c.max(1).indices
    nn_yx = c.max(0).indices
    keep = nn_yx[nn_xy] == torch.arange(len(xq), device=s.device)
    return torch.arange(len(xq), device=s.device)[keep], nn_xy[keep]


def plane_rotate(Q, u, v, theta):
    """Right-multiply Q by expm(theta*(u v^T - v u^T)) using the rank-2 form.

    For orthonormal u,v: expm(tS) = I + sin(t) S + (1-cos(t)) S^2, and
    S^2 = -(u u^T + v v^T). Apply as Q' = Q @ expm(theta S) (rotation acting
    in input coordinates of Q^T... we act on the OUTPUT side: Q' = R_theta Q).
    """
    s, c1 = math.sin(theta), 1.0 - math.cos(theta)
    Qu = Q.T @ u  # (d,)
    Qv = Q.T @ v
    # R Q = Q + s (u v^T - v u^T) Q + c1 * (-(u u^T + v v^T)) Q, acting from the left
    return Q + s * (torch.outer(u, Qv) - torch.outer(v, Qu)) - c1 * (torch.outer(u, Qu) + torch.outer(v, Qv))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_a", required=True)
    p.add_argument("--emb_b", required=True)
    p.add_argument("--emb_b_train", default="")
    p.add_argument("--init", choices=["random", "perturbed_truth"], default="random")
    p.add_argument("--perturb_angle", type=float, default=2.0)
    p.add_argument("--sig_dim", type=int, default=128)
    p.add_argument("--move_space", choices=["signal", "full"], default="signal")
    p.add_argument("--n_sub", type=int, default=8000)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--n_steps", type=int, default=30000)
    p.add_argument("--t0", type=float, default=0.02, help="initial temperature (energy units; 28-sigma gap ~ 1.0)")
    p.add_argument("--t_end", type=float, default=0.0005)
    p.add_argument("--step0", type=float, default=0.6, help="initial plane-rotation angle scale (rad)")
    p.add_argument("--detector_thresh", type=float, default=3.0,
                   help="ICP handoff when mutual count > thresh x baseline")
    p.add_argument("--icp_iters", type=int, default=40)
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
    n = len(rest) // 2
    ia, ib = rest[:n], rest[n:2 * n]
    Xev, Yev = ea[ie].to(device), eb[ie].to(device)
    Xfull = ea[ia].to(device)
    if args.emb_b_train:
        Yfull = normalize(torch.load(args.emb_b_train, map_location="cpu", weights_only=True))[:n].to(device)
    else:
        Yfull = eb[ib].to(device)
    X, Y = Xfull[:args.n_sub], Yfull[:args.n_sub]

    Wsup, _ = orthogonal_procrustes(ea[ia].numpy(), eb[ia].numpy())
    Qsup = torch.from_numpy(Wsup.T).float().to(device)
    rank_sup = mean_rank(normalize(Xev @ Qsup.T), Yev)

    # signal subspace of the TARGET cloud (move planes live here)
    Yc = (Yfull - Yfull.mean(0, keepdim=True)).cpu()
    V_sig = torch.linalg.svd(Yc, full_matrices=False).Vh[:args.sig_dim].to(device)

    if args.init == "random":
        G = torch.randn(d, d, generator=torch.Generator().manual_seed(args.seed + 7))
        Qr, Rr = torch.linalg.qr(G)
        Q = (Qr * torch.sign(torch.diagonal(Rr))).to(device)
        if torch.det(Q).sign() != torch.det(Qsup).sign():
            Q[0] = -Q[0]
    else:
        P = torch.randn(d, d, generator=torch.Generator().manual_seed(args.seed + 7))
        skew = P - P.T
        skew = skew / skew.norm() * args.perturb_angle * (d ** 0.5)
        Q = (torch.matrix_exp(skew) @ Qsup).to(device)

    d_xx = cross_mean_dist(X, X)
    d_yy = cross_mean_dist(Y, Y)

    def L(Qc):
        return 2 * cross_mean_dist(X @ Qc.T, Y) - d_xx - d_yy

    L_true = L(Qsup)
    Lc = L(Q)
    rank_init = mean_rank(normalize(Xev @ Q.T), Yev)
    # detector baseline: mutual count for a random rotation
    base_mutual = csls_mutual_count(normalize(X[:4000] @ Q.T), Y[:4000])
    print(f"init L {Lc:.4f} (true {L_true:.4f}), rank {rank_init:.1f}, mutual baseline {base_mutual}", flush=True)

    gmv = torch.Generator(device="cpu").manual_seed(args.seed + 13)
    best_L, best_Q = Lc, Q.clone()
    accepted = 0
    fired = False
    traj = []
    for step in range(1, args.n_steps + 1):
        frac = step / args.n_steps
        T = args.t0 * (args.t_end / args.t0) ** frac
        stepsize = args.step0 * (0.05 / args.step0) ** frac
        # random plane in the signal subspace (or full space)
        if args.move_space == "signal":
            c = torch.randn(2, args.sig_dim, generator=gmv).to(device) @ V_sig
        else:
            c = torch.randn(2, d, generator=gmv).to(device)
        u = F.normalize(c[0], dim=0)
        v = c[1] - (c[1] @ u) * u
        v = F.normalize(v, dim=0)
        theta = float(torch.randn(1, generator=gmv)) * stepsize
        Qp = plane_rotate(Q, u, v, theta)
        Lp = L(Qp)
        if Lp < Lc or math.exp(-(Lp - Lc) / max(T, 1e-9)) > float(torch.rand(1, generator=gmv)):
            Q, Lc = Qp, Lp
            accepted += 1
            if Lc < best_L:
                best_L, best_Q = Lc, Q.clone()
        if step % 2000 == 0:
            # re-orthogonalize (float32 drift over many rank-2 updates)
            U_, _, Vh_ = torch.linalg.svd(Q)
            Q = U_ @ Vh_
            Lc = L(Q)
        if step % 1000 == 0:
            r = mean_rank(normalize(Xev @ Q.T), Yev)
            mut = csls_mutual_count(normalize(X[:4000] @ Q.T), Y[:4000])
            traj.append((step, round(Lc, 4), round(r, 1), mut, round(T, 5)))
            print(f"step {step}: L {Lc:.4f} rank {r:.1f} mutual {mut} T {T:.5f} acc {accepted/step:.2f}", flush=True)
            if mut > args.detector_thresh * max(base_mutual, 20):
                print("DETECTOR FIRED -> ICP handoff", flush=True)
                fired = True
                break

    # ICP handoff from best-L state (or current if detector fired)
    Qh = Q if fired else best_Q
    rank_pre_icp = mean_rank(normalize(Xev @ Qh.T), Yev)
    sub = min(len(Xfull), len(Yfull), 20000)
    Xs, Ys = Xfull[:sub], Yfull[:sub]
    for it in range(args.icp_iters):
        with torch.no_grad():
            pa, pb = csls_mutual(normalize(Xs @ Qh.T), Ys)
            if len(pa) < 50:
                break
            Wi, _ = orthogonal_procrustes(Xs[pa].cpu().numpy(), Ys[pb].cpu().numpy())
            Qh = torch.from_numpy(Wi.T).float().to(device)
    r_final = mean_rank(normalize(Xev @ Qh.T), Yev)
    print(f"pre-ICP rank {rank_pre_icp:.1f} -> final {r_final:.1f}", flush=True)

    result = {"emb_a": args.emb_a, "emb_b": args.emb_b, "emb_b_train": args.emb_b_train,
              "init": args.init, "perturb_angle": args.perturb_angle,
              "move_space": args.move_space, "sig_dim": args.sig_dim, "seed": args.seed,
              "n_steps": args.n_steps, "t0": args.t0, "step0": args.step0,
              "rank_supervised": rank_sup, "rank_init": rank_init,
              "L_true": L_true, "L_best": best_L, "detector_fired": fired,
              "rank_pre_icp": rank_pre_icp, "rank_final": r_final, "trajectory": traj,
              "verdict": "SOLVED" if r_final < 5 * max(rank_sup, 1) else ("PARTIAL" if r_final < 100 else "FAIL")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory"}))


if __name__ == "__main__":
    main()
