# Fable fleet 1: global-optimization reframe of the no-requery conjunction cell

Date: 2026-07-06. Follows `v3.md` (main checkout). Branch:
`worktree-fable-identifiability-experiments`.

## The reframe motivating this fleet

v3's own B1 + B1b results establish that the energy-distance objective's
optimum sits AT R_true on the conjunction cell (static margin ~0.8 normalized
vs random rotations; descent from truth stays at rank 1.6). That is
**identifiability, demonstrated** — the "barrier" framing is contradicted by
the record. The open problem is **global optimization of a verified-aligned
objective**, which has a standard toolbox never applied here: graduated
non-convexity, batch scaling (gradient-noise control), higher-order-moment
initialization, preconditioning.

## Record corrections found while building this fleet (fold into v3.md §0)

1. **`drift_deg_from_truth` in `objective_descent.py` is a null metric.** It
   takes SVD singular values of M = Q^T Q_sup, which is orthogonal, so they are
   identically 1 and arccos reads ~0 regardless of drift. The ledger's
   "SWD drifts only 0.7 deg yet rank collapses" narrative is unsupported (the
   rank numbers stand; the angle characterization is float noise). Correct
   metric = complex eigenvalue phases of M (implemented in this fleet's
   scripts). The "moves in exactly the retrieval-critical subspace" claim
   should be re-derived with the fixed metric.
2. **`objective_descent.py` does not log `--emb_b_train`** — the conj runs are
   only verifiable via the Slurm `Command:` echo (I checked: they did use
   FineWeb-gtr). Fixed in this fleet's scripts.
3. **v2's wide (1.5 rad) ICP basin is a SAME-DATASET measurement**
   (`out/orth/icpbasin_*` all use NQ-gtr; `orthogonal_align.py` has no
   cross-corpus support). The localization claim "only the coarse matcher is
   missing" rests on an unmeasured assumption for the conjunction cell —
   experiment E1 fills it.
4. **Scale confound on the decisive gate:** `ganconj_*` runs at 100k/side, but
   the record shows GAN-at-100k fails an easier cross-dataset cell
   (stella-gte: stuck ~307) that GAN-at-1M solves (65-101). A ganconj failure
   at 100k does not establish the barrier.

## Fleet (pre-registered predictions)

All cells: A = gte-NQ; paired eval on NQ (pool 4096); conjunction train cloud =
FineWeb-gtr (`out/emb_fineweb/gtr.pt` in the main checkout). Perturbation
seeding identical to `ebasin_*` (seed+7) so init ranks are comparable.

### E1. ICP basin on the conjunction cell (`scripts/icp_basin.py`)

Cells: conj pert {0.5, 1.0, 1.5} + same-dataset pert 1.5 (reimplementation
check vs v2's icpbasin_1.5 = rank 2.7).

**Prediction:** ICP is the most assignment-based refiner (hard mutual-NN
pairs); per the assignment-free principle it mis-couples under support
mismatch. Conjunction basin well below 1.5 rad; plausibly drifts even from
0.5. If instead STABLE at 1.5, the v2 localization survives and the
coarse-matcher budget stays 1.5 rad.

### E2. Big-batch energy descent (`scripts/energy_descent2.py --accum 16`)

Cells: conj pert {1.0, 1.5}, effective batch 32k (vs baseline 2048), 8k steps.

**Prediction:** if the 1.5-rad stall (~250 plateau) is minibatch gradient
noise (H-noise), big-batch lands well below 250 (and pert-1.0 goes below its
8.4). If unchanged, the stall is landscape-structural (H-landscape) and GNC
(E3) is the needed fix.

### E3. GNC / annealed-noise energy descent (`--sigma_start 0.5 --sigma_end 0.01`)

Cells: conj pert {1.5, 2.0} (accum 4, 20k steps) + random-init moonshot
(30k steps).

**Prediction:** annealed smoothing widens energy's basin beyond 0.5 rad:
pert-1.5 ends < 50 (baseline 261). Pert-2.0 informative either way. Random
init likely still fails (volume of SO(768)) but its trajectory shape tells us
whether smoothing creates any long-range signal.

### E4. ICA fourth-order coarse matcher (`scripts/ica_align.py`)

First method in the program using statistics above second order — the
isotropy wall (GW, cov-axes, spectral, QAP all fail on it) is a second-order
statement; ICA identifiability (Comon) survives isotropic covariance given
non-Gaussianity, and embedding clouds are cluster-structured (non-Gaussian).
Lit support: ICA axes correspond across independently trained embedding
models (Yamagiwa et al. 2023, arXiv:2305.13175).

Cells: gte-e5 (within-lineage POSITIVE CONTROL — gate), gte-gtr NQ
(same-dataset cross-lineage — the real test), conjunction. k=128, 3 ICA
restarts/side, confidence-gated axis matching.

**Prediction (honest priors):** control passes (rank < 100 after ICP) with
~0.5 prob; same-dataset cross-lineage inside a refiner basin would be a
first for correspondence-free methods. Synthetic smoke (hostile: exchangeable
clusters with near-identical marginals) shows partial axis correspondence
(10/24 oracle) and matcher precision ~30%, improved by restarts+gating; real
embedding axes are known to be more distinctive. If the control fails, the
method is disqualified per program discipline (but log WHERE: axis
correspondence vs matching, via `matched_axis_w1` and `confident_axes`).

## Verdict semantics

Same as objective_descent: STABLE < 5x sup, PARTIAL < 100, else DRIFTS/FAIL.

## Results

(fill as jobs land; job table below)
