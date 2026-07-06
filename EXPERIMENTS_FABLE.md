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

### E1 (2026-07-06, jobs 767673-6): PREDICTION REFUTED — ICP basin survives
the conjunction cell

| cell | pert (rad) | init rank | final rank | verdict |
|---|---|---|---|---|
| same-dataset | 1.5 | 392 | **2.7** | STABLE (reproduces v2 icpbasin_1.5 exactly — reimpl validated) |
| conjunction | 0.5 | 1.0 | **1.2** | STABLE |
| conjunction | 1.0 | 3.1 | **1.5** | STABLE (converges TO truth) |
| conjunction | 1.5 | 392 | **41.6** | PARTIAL (vs 2.7 same-dataset; vs energy's 261 from same init) |

Delta to model: the assignment-free principle does NOT extend to ICP —
CSLS mutual-NN pairs across different corpora are good-enough
correspondences (semantically-nearest FineWeb neighbors of NQ points are
approximately correct pairs, unlike SWD's monotone sort or Sinkhorn's
balanced plan). **v2's localization claim survives on the conjunction cell:
the coarse matcher budget is ~1.0-1.5 rad, and ICP remains the finisher.**
ICP also dominates energy as a refiner from deep inits (41.6 vs 261 from
1.5 rad).

Metric note: the STABLE solutions sit at 70-83 deg MEAN eigenphase from
R_true while scoring rank ~1 — the rotation is only pinned on the ~128-d
signal subspace; the complement is free. Mean-eigenphase drift is therefore
also a misleading summary (as was the null SVD metric); drift should be
reported restricted to the shared signal subspace.

### E2 (jobs 767677/8): the stall is NOT simple gradient noise — it is a
transient pass NEAR truth followed by drift to a displaced optimum

| cell | pert | eff. batch | trajectory | final |
|---|---|---|---|---|
| conj | 1.0 | 32k | init 3.1 -> **min 1.9 @ step 250** -> rises | 8.0 (baseline 8.4) |
| conj | 1.5 | 32k | init 392 -> **min 134.5 @ step 2000** -> rises | 243.7 (baseline 261) |

Delta to model: with 16x larger batches the non-monotone shape persists →
the empirical energy optimum on these finite mismatched clouds is DISPLACED
from R_true, but the descent PATH passes near truth before settling there.
End-of-descent rank is the wrong deliverable; the trajectory minimum is —
and 134 is inside ICP's conjunction basin (E1: ICP recovered from init 392).
**Actionable: chain = short big-batch energy descent -> checkpoint by the
validated unsupervised CSLS criterion -> ICP.** Fleet 2 (jobs 767750-4)
tests this from 1.5/2.0/2.5 rad with ICP-only controls.

### E3 (jobs 767679-81): GNC/annealed smoothing REFUTED — noise destroys
the signal instead of revealing a coarse basin

pert 1.5: 392 -> 811 (rises steadily during high sigma, never recovers);
pert 2.0: 2402 -> 2196 (stuck); random init: 2063 -> 2019 (stuck).

Delta to model: there is NO coarse-scale basin to exploit — Gaussian
smoothing makes the clouds more isotropic, which is exactly the degeneracy.
ALL the alignment signal is fine-scale. This deepens the isotropy-wall
picture (it is a statement about coarse/smoothed views in general, not just
second-order statistics) and explains why the GAN's coarse-to-fine works
where blurring does not: adaptive witness features are feature SELECTION,
not isotropic smoothing.

### E4 (jobs 767682-4): ICA matcher FAILS its within-lineage control —
disqualified as implemented

gte-e5 control: coarse 1401, after ICP 1051 (sup 1.0) → FAIL. Same-dataset
1712, conjunction 2246. Matched-marginal costs are similar across cells
(~0.025-0.033), i.e. the matcher cannot distinguish right from wrong axis
pairings. Synthetic oracle diagnostic (smoke) showed FastICA sample
instability + PCA-tail mismatch + weak marginal signatures each contribute.
The fourth-order route is not fundamentally refuted (a joint
cumulant-tensor alignment could still work where per-axis matching fails),
but per program discipline this implementation is dead. Prior lowered.

### E5 / fleet 2 (jobs 767750-4): chain (energy -> CSLS checkpoint -> ICP)
FAILS — two separable causes

| run | init | energy end | CSLS-selected step | rank there | chain final |
|---|---|---|---|---|---|
| chain 1.5 | 392 | 205 | 6000 (= end) | 205 | 253 |
| chain 2.0 | 2402 | 2363 | 6000 (= end) | 2363 | 2260 |
| chain 2.5 | 3237 | 3353 | 6000 (= end) | 3353 | 3005 |
| icp-only 2.0 | 2401 | — | — | — | 2280 (DRIFTS) |
| icp-only 2.5 | 3237 | — | — | — | 2911 (DRIFTS) |

Two findings:
1. **The unsupervised CSLS criterion is monotone along the descent** (rises
   0.007 -> 0.017 while rank passes through its dip and back up), so it
   selects the ENDPOINT, never the dip. It measures distributional coupling,
   not registration — consistent with v2's "criterion detects solved runs,
   cannot rank partial ones". No unsupervised dip-detector yet.
2. **Equal rank does NOT mean equal ICP-basin membership.** ICP recovers
   from a rank-392 random perturbation of truth (E1: -> 41.6) but NOT from
   the rank-205 energy-drifted rotation (-> 253). Energy's drift
   concentrates error in the retrieval-critical directions (a weaker
   version of the SWD pathology); random perturbations spread it
   isotropically. Basin membership is direction-dependent, not
   rank-dependent — direction-resolved geometry (Hessian spectrum, P2
   queue) is the right next diagnostic.

Also: ICP-only controls confirm ICP's conjunction basin ends between 1.5
and 2.0 rad.

### E5b (jobs 767928/9, oracle ceiling — measuring instrument): is the
trajectory DIP inside ICP's basin?

Oracle-select the best-rank checkpoint (instead of CSLS) and run ICP from
it. If the dip point converges -> the chain concept survives and ONLY
unsupervised dip-selection is missing. If not -> the energy path never
enters ICP's basin and the chain is dead.

**RESULT (767928): the chain is DEAD — even the oracle-selected dip is
poorly ICP-recoverable.** From 1.5 rad: dip at step 1750 (rank 134) -> ICP
-> 71.5. Compare ICP-alone from the RAW 1.5-rad init (rank 392) -> 41.6.
The energy stage makes basin membership WORSE even while improving rank:
its descent concentrates the residual error into the retrieval-critical /
ICP-pathological directions. Rank is not a basin coordinate; direction of
error is. (2.0-rad twin 767929 CONFIRMED the pre-registered expectation:
no dip exists from 2.0 rad — init 2402, best 2389 at step 750, ICP 2327.)

Delta to model: energy descent is only useful ON-basin (<= 0.5 rad, where it
is stable at truth); as a mid-range transporter it actively harms. The
surviving recipe is unchanged from E1: ANY coarse init within ~1.0-1.5 rad
-> ICP. The entire program still reduces to the coarse matcher, but E1
showed the target is more forgiving than feared (ICP tolerates cross-corpus
mutual-NN pairs). Highest-value next diagnostics: direction-resolved
Hessian/error analysis (WHY are energy-drifted errors ICP-pathological?),
and the P2 queue (1M gates, unbalanced OT, consistency-graph pair
distillation).
