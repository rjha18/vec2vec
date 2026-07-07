# No-Requery Embedding Alignment — Full Progress Writeup

Date: 2026-07-07. Branch `worktree-fable-identifiability-experiments`,
PR rjha18/vec2vec#22. Standalone summary of the Fable-session program
(follows `v3.md`; per-experiment ledger with commands/job IDs is in
`EXPERIMENTS_FABLE.md`). Written as a resumable handoff.

---

## 0. The problem, in one paragraph

Two frozen, unpaired embedding clouds: A = gte on NQ, B = gtr on FineWeb.
NO REQUERY — never run either encoder on new text (anchors, supervised
Procrustes R_true, and re-encodings are MEASURING INSTRUMENTS ONLY, never
part of a method). Question: recover the map A->B from the two marginals
alone. Prior work localized this to ONE open cell — the **conjunction**:
cross-lineage (gte is BERT-family, gtr is T5-family, ~90 deg apart) AND
cross-dataset (NQ vs FineWeb, disjoint supports). Success = held-out paired
retrieval rank <= ~15 of a 4096 pool (random ~2000).

Inherited facts (v2/v3, all reproduced): the true map is a FIXED orthogonal
rotation, a property of the encoder PAIR not the data (transfers NQ->FineWeb
at rank 1.02, 0 deg apart); cross-lineage that rotation is a generic ~90 deg
full-dimensional rotation; alignment lives in a ~128-d shared subspace;
~500 supervised pairs solve it (bridge sweep: N=100->124, 200->20, 500->1.7).

---

## 1. Headline results of this session

1. **It is NOT an identifiability barrier — it is a search problem.** The
   energy-distance objective's global optimum sits at R_true even
   cross-dataset, with a ~28-sigma margin over random rotations, and that
   margin is FLAT in sample size (N=5k..122k). Identifiability is a settled
   population property; the difficulty is optimization/search.

2. **The alignment currency is ~500 effective correspondences ("tethers"),
   however obtained.** Supervised pairs, ICP-generated mutual-NN tethers,
   and oracle-selected cross-corpus proxies all obey the same ~200-500 knee.
   ICP is a tether bootstrap; it self-sustains iff its noisy tether ensemble
   carries ~500 effective-good correspondences (count x precision).

3. **The oracle tether ceiling SOLVES the conjunction cell (rank ~2).** With
   the best cross-corpus proxies selectable in principle, ~500 tethers ->
   rank 9 (Procrustes) -> 2.2 (ICP); 5000 -> rank 1. So the cell is
   information-theoretically solvable from the marginals; the ONLY missing
   piece is selecting good tethers without the oracle. (See the open gap, §4.)

4. **The difficulty has a named mechanism: the entropic trap.** All paths
   from random rotations to truth are perfectly monotone in energy (geodesic
   probe), yet descent from >= 2.0 rad NEVER improves rank at any batch size:
   the local gradient is registration-blind at long range — it spends itself
   on registration-free shape/moment matching. Only a mechanism that
   RESHAPES local geometry toward the registration direction (an adaptive
   discriminator) can escape; isotropic mechanisms (smoothing, transport,
   value-SA with random moves) cannot.

5. **UNE corroboration.** Text embedding clouds are ~93% Gaussian in random
   1-d projections (everywhere, incl. the signal subspace) — nearly
   maximally rotationally symmetric. Alignability lives in a thin, diffuse,
   partly encoder-specific non-Gaussian residual. This is the sharpest
   one-line cause of the whole negative-space table and gives strong
   out-of-domain support to arXiv:2603.21786 (Universal Normal Embedding).

---

## 2. What is solved / what is open (localization)

SOLVED:
- The map class (orthogonal, fixed, dataset-independent).
- Identifiability from marginals (28-sigma margin, flat in N).
- The finisher: ICP converges to rank ~1 from any init within ~1.0-1.5 rad,
  and it TOLERATES cross-corpus tethers (E1: conj 1.0 rad -> 1.5, 1.5 rad ->
  41.6; same-dataset 1.5 rad -> 2.7). Reimplementation validated vs v2.
- The refiner discipline: energy distance is the UNIQUE objective stable at
  truth cross-dataset; every objective with a coupling/assignment mechanism
  (SWD, balanced + unbalanced Sinkhorn, ICP-beyond-basin) mis-registers
  under support mismatch.

OPEN (the entire remaining problem):
- A cold-start coarse matcher that reaches the ~1.5-rad ICP basin from ~90
  deg, unsupervised. Equivalently: select ~500 good tethers without the
  oracle. Every method tried fails (§3).

---

## 3. The negative-space map (methods tried, all failed, with mechanisms)

| Method | Result | Mechanism of failure |
|---|---|---|
| QAP / GW / cov-axes / spectral coarse | ~2000 | isotropy wall (no canonical 2nd-order axes) |
| Point descriptors, consistency-graph | ~2000 (control fail) | distance concentration in 768-d |
| ICA / 4th-order axis matching | control fail | FastICA sample-instability; marginals indistinctive |
| Marginal reweighting (oracle+blind) | ~2000 | marginals already coincide at truth |
| Random-restart SO(d) + CSLS select | control fail | volume of SO(768); criterion can't rank partials |
| SWD / Sinkhorn / unbalanced-OT descent | drifts | coupling mis-registers under support mismatch |
| Energy descent (any batch), from >=1.5 rad | stalls | entropic trap (registration-blind gradient) |
| GNC / annealed smoothing | worse | smoothing increases isotropy = the degeneracy |
| Energy->ICP chain (CSLS-selected dip) | fails | CSLS blind to the dip; energy drift is ICP-pathological |
| funnel-SA (value Metropolis, random moves) | fails | random proposals pay 1/dim(SO(128)) |
| scorepot (spconj) on conjunction | dead (multi-seed) | persistent field surmounts lineage OR support, not both |
| GAN (ganconj) on conjunction, 100k | dead (2 seeds, ~random) | 2 seeds != verdict; scale/lottery unresolved |

Unifying statement: every STATIC, hand-designed functional of the marginals
fails in exactly the ~128 directions that carry retrieval signal, because
those directions are where the near-Gaussian clouds are least discriminative.

---

## 4. The current frontier: tether identification

The oracle ceiling (§1.3) says the cell is solvable IF we can select ~500
good tethers unsupervised. Two sub-questions, both now measured:

**(a) How destructive are false-positive tethers? (fp_robustness.py)**
The precision bar is LOW at scale, and depends on the estimator:
- K=5000 tethers: raw Procrustes tolerates 50% FP (rank 2.4); ICP tolerates
  ~50%; breaks by 70-90%.
- K=1000: raw breaks earlier (30% FP -> rank 10), but raw+ICP holds to ~50%.
- Trimmed (robust) Procrustes helps most at small-K/high-FP.
Bar: a selector needs only ~40-50% precision at K~1000-5000. Not stringent.

**(b) What unsupervised signal identifies good tethers? (tether_signals.py)**
Scored csls_margin / gap12 / neg_density / cycle / nn_overlap vs the oracle,
at perturb 0.5 / 1.0 / 1.5 rad. THE SIGNALS EVAPORATE WHERE NEEDED:
- 0.5 rad (baseline precision 67%): all signals lift to 85-94% — easy.
- 1.0 rad (baseline 46%): csls_margin & nn_overlap lift to ~66% — usable.
- 1.5 rad (baseline 6%, the basin-edge regime): best (csls_margin) lifts top
  decile to only 10%; nn_overlap 7%. BELOW the FP-tolerance bar.
So near the basin everything works; far from it, tethers are too wrong to
select from. This is the gap in one picture: no signal bridges the
6%-precision regime a cold start sits in.

Note: v2's earlier claim "CSLS is blind" was tether-LAW specific (selecting
among already-converged tethers); here csls_margin DOES carry rank
information at moderate init — the two are consistent (it correlates with
correctness only once you are partway in).

---

## 5. The assembled cold-start method (running now: jobs 778532-4)

`tether_method.py`: R random restarts -> trimmed-Procrustes tether bootstrap
each -> select best by unsupervised mean-CSLS criterion. This is the honest
cold-start counterpart to the oracle ceiling — the actual deployable method
the week's components imply. Prior: likely FAILS (random restarts don't land
in-basin in SO(768); signal collapse at >=1.5 rad). Value: (i) the real
method NUMBER rather than component measurements; (ii) the criterion-vs-rank
scatter shows whether SELECTION would work if any restart ignited. Result
pending; will be appended to EXPERIMENTS_FABLE.md.

HONEST STATUS: no fully-unsupervised method (this one, GAN, scorepot,
funnel-SA, coarse matchers) has solved the conjunction cell from a cold
start. Everything at rank ~1-40 used the oracle or a label-derived
(perturbed-truth) init. The cell remains open.

---

## 6. Record corrections found this session (fold into v3.md)

1. `drift_deg_from_truth` (objective_descent.py) is a NULL metric: SVD
   singular values of the orthogonal M=Q^T Qsup are identically 1, so it
   reads ~0 regardless of drift. v2's "SWD drifts only 0.7 deg" is a metric
   artifact; the direction claim was right under the corrected (eigenphase)
   metric. Even mean-eigenphase misleads: rank-1 solutions sit 70-83 deg
   from R_true because only the 128-d signal subspace is pinned.
2. `emb_b_train` was not logged in descent JSONs (conj runs verified correct
   via Slurm command echoes; fixed in new scripts).
3. v2's wide (1.5 rad) ICP basin was a SAME-DATASET measurement; E1 filled
   the conjunction-cell gap (it holds).
4. ganconj gate runs at 100k/side where the GAN is known to fail easier
   cross-dataset cells that 1M solves — a 100k negative is not a barrier
   verdict.

---

## 7. Theoretical picture (for the paper spine)

Identifiability theorem shape (B3): R is identified by the target marginal
iff that marginal has trivial orthogonal symmetry group on the shared signal
subspace, with margin gamma = curvature of the divergence at truth.
Measured: gamma is large (28-sigma), flat in N, concentrated in the ~128-d
signal subspace (curvature there ~1000x the complement). The clouds are
~93% Gaussian (UNE), i.e. near-maximally symmetric; identifying information
is a thin non-Gaussian residual. Hence: identified-but-unfindable — the
information exists (oracle ceiling solves) but no local or static method
concentrates on the residual, and the ~500-tether currency quantifies the
cost. Only an adaptive adversarial witness is conjectured to reshape local
geometry onto the residual (Q2, untested cleanly — the live question).

---

## 8. Next steps (resumable)

Ranked by information-per-GPU-hour:
1. **Harvest jobs 778532-4** (cold-start method number) — the honest current
   answer.
2. **GAN mechanism dissection** (the one survivor of the negative-space
   table). Under the convergence protocol: >=8 seeds, prune by the validated
   rule (rank<300 ever = ignited; >480 at 100k = dead), pin 2080ti, select
   by CSLS criterion. Instrument the WORKING GAN (gan2080, saves
   discriminators) with the direction-resolved lens: do D's input-gradients
   concentrate in the signal subspace early? Does the translator's error
   stay direction-benign (unlike energy descent)? This directly tests the
   entropic-trap escape hypothesis. C1 (discriminator-replay) + C2 (lagged-D
   sweep) decide min-max essentiality.
3. **GAN at partial support overlap** (f=0.5/0.75, motivated by the
   dose-response curve): isolates what adversarial dynamics buy over QAP+ICP
   as a function of overlap, on cells where ignition is plausible.
4. **Better tether signals for the 6%-precision regime** (the real gap):
   pairwise geometric consistency, cluster-level agreement, persistence
   across iterations — signals independent of the matcher's own similarity,
   which is registration-blind far from truth.
5. **Characterization paper** — the spine (§7) is measured and strong
   regardless of whether the cell falls: identified-but-unfindable, the
   ~500-tether currency, the entropic-trap mechanism, the negative-space
   map, UNE corroboration.

## Scripts written this session (all in scripts/, on the PR)
icp_basin.py (E1 + tether precision/selection), energy_descent2.py
(big-batch/GNC/unbalanced-OT/chain), ica_align.py, direction_resolved.py,
consistency_match.py, margin_vs_n.py, gaussianity_probe.py (UNE),
geodesic_probe.py, funnel_sa.py, oracle_tether_ceiling.py, fp_robustness.py,
tether_signals.py, tether_method.py. Encode: 250k NQ (gte,gtr) + 250k
FineWeb-gtr caches (out/emb_nq_250k, out/emb_fineweb_250k).
