from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


FmriFormat = Literal["avg", "trial"]


@dataclass(frozen=True)
class FmriSplitConfig:
    data_dir: str
    fmt: FmriFormat
    seed: int = 42
    train_on_overlap: bool = False
    # avg: number of paired test samples in each split
    avg_train_size: int = 500
    avg_val_size: int = 500
    avg_eval_size: int = 1000
    # trial: split by unique stimulus labels (each contributes 3 repetitions)
    trial_train_images: int = 500
    trial_val_images: int = 500
    trial_eval_images: int = 1000


class FmriSingleViewDataset(Dataset):
    """A dataset that returns a single named embedding per example.

    Each item is `{view_name: Tensor(d,)}` so that the existing training loop
    can merge two dataloaders (sup + unsup) without calling `process_batch`.
    """

    def __init__(self, view_name: str, Z: torch.Tensor):
        if Z.ndim != 2:
            raise ValueError(f"Expected Z to be 2D (n,d). Got shape={tuple(Z.shape)}")
        self.view_name = view_name
        self.Z = Z

    def __len__(self) -> int:
        return int(self.Z.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {self.view_name: self.Z[idx]}


class FmriPairedDataset(Dataset):
    """A paired dataset that returns both named embeddings per example."""

    def __init__(self, name_a: str, Za: torch.Tensor, name_b: str, Zb: torch.Tensor):
        if Za.ndim != 2 or Zb.ndim != 2:
            raise ValueError("Za/Zb must be 2D (n,d)")
        if Za.shape != Zb.shape:
            raise ValueError(f"Za and Zb must match. Got Za={Za.shape}, Zb={Zb.shape}")
        self.name_a = name_a
        self.name_b = name_b
        self.Za = Za
        self.Zb = Zb

    def __len__(self) -> int:
        return int(self.Za.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {self.name_a: self.Za[idx], self.name_b: self.Zb[idx]}


def _resolve_data_dir(data_dir: str) -> str:
    return os.path.abspath(os.path.expanduser(data_dir))


def _load_subject_checkpoint(data_dir: str, template: str, subject: int) -> Dict[str, torch.Tensor]:
    path = os.path.join(data_dir, template.format(subject=subject))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find fMRI embeddings file at {path}. "
            f"Set `fmri_data_dir` and/or adjust template."
        )
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected a dict checkpoint in {path}")
    return ckpt


def _randperm(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randperm(int(n), generator=g)


def _normalize_rep(rep: int) -> int:
    # Notebook uses repetitions starting at 0 then adds 1; accept both.
    if rep in (0, 1, 2):
        return rep + 1
    if rep in (1, 2, 3):
        return rep
    raise ValueError(f"Unexpected repetition id {rep}. Expected 0-2 or 1-3.")


def _trial_triplets(labels: torch.Tensor, repetitions: torch.Tensor) -> Dict[int, Tuple[int, int, int]]:
    """Return {stimulus_label: (idx_rep1, idx_rep2, idx_rep3)}."""
    if labels.shape != repetitions.shape:
        raise ValueError("labels and repetitions must match shape")
    triplets: Dict[int, List[int]] = {}
    seen: Dict[Tuple[int, int], int] = {}
    for i in range(int(labels.shape[0])):
        lab = int(labels[i].item())
        rep = _normalize_rep(int(repetitions[i].item()))
        key = (lab, rep)
        if key in seen:
            raise ValueError(f"Duplicate (label,rep) encountered: {key}")
        seen[key] = i
        if lab not in triplets:
            triplets[lab] = [None, None, None]  # type: ignore[list-item]
        triplets[lab][rep - 1] = i  # type: ignore[index]

    out: Dict[int, Tuple[int, int, int]] = {}
    for lab, idxs in triplets.items():
        if any(v is None for v in idxs):
            raise ValueError(f"Label {lab} is missing one or more repetitions")
        out[int(lab)] = (int(idxs[0]), int(idxs[1]), int(idxs[2]))  # type: ignore[arg-type]
    return out


def _indices_from_labels(triplets: Dict[int, Tuple[int, int, int]], chosen_labels: Sequence[int]) -> np.ndarray:
    # Deterministic order: sort labels, then reps 1..3.
    idxs: List[int] = []
    for lab in sorted(int(x) for x in chosen_labels):
        r1, r2, r3 = triplets[int(lab)]
        idxs.extend([r1, r2, r3])
    return np.asarray(idxs, dtype=np.int64)


def load_fmri_paired_splits(
    *,
    cfg: FmriSplitConfig,
    sup_name: str,
    unsup_name: str,
    sup_subject: int = 1,
    unsup_subject: int = 2,
) -> Tuple[FmriPairedDataset, FmriPairedDataset, int]:
    """Load paired train/val datasets from precomputed fMRI embeddings.

    Returns (train_paired, val_paired, embed_dim).
    """
    data_dir = _resolve_data_dir(cfg.data_dir)
    if cfg.fmt == "avg":
        template = "avg_ws_mlp_v1_128_768_sub-{subject:02d}.pt"
    elif cfg.fmt == "trial":
        template = "trial_ws_mlp_v1_128_768_sub-{subject:02d}.pt"
    else:
        raise ValueError(f"Unknown fmt: {cfg.fmt}")

    sup_ckpt = _load_subject_checkpoint(data_dir, template, sup_subject)
    unsup_ckpt = _load_subject_checkpoint(data_dir, template, unsup_subject)

    # Shared test set is aligned across subjects; we only use it for paired supervision.
    Zs = sup_ckpt["Z_test"].to(torch.float32)
    Zu = unsup_ckpt["Z_test"].to(torch.float32)
    if Zs.shape != Zu.shape:
        raise ValueError(f"Subject test embeddings mismatch: {Zs.shape} vs {Zu.shape}")
    embed_dim = int(Zs.shape[1])

    if not torch.equal(sup_ckpt["labels_test"], unsup_ckpt["labels_test"]):
        raise ValueError("labels_test must match across subjects for paired training/eval.")

    if cfg.fmt == "avg":
        n = int(Zs.shape[0])
        perm = _randperm(n, cfg.seed).numpy()
        need = int(cfg.avg_train_size) + int(cfg.avg_val_size)
        if need > n:
            raise ValueError(f"Need train+val={need} but only n={n} test samples exist.")
        train_idx = perm[: int(cfg.avg_train_size)]
        val_idx = perm[int(cfg.avg_train_size) : need]
    else:
        # Split by unique stimulus labels, keeping all 3 repetitions per stimulus.
        reps_s = sup_ckpt.get("repetitions_test", None)
        reps_u = unsup_ckpt.get("repetitions_test", None)
        if reps_s is None or reps_u is None:
            raise ValueError("trial format requires repetitions_test in both checkpoints.")
        if not torch.equal(reps_s, reps_u):
            raise ValueError("repetitions_test must match across subjects for paired trial splits.")

        triplets = _trial_triplets(sup_ckpt["labels_test"], reps_s)
        unique_labels = np.asarray(sorted(triplets.keys()), dtype=np.int64)
        rng = np.random.default_rng(int(cfg.seed))
        rng.shuffle(unique_labels)
        need_imgs = int(cfg.trial_train_images) + int(cfg.trial_val_images)
        if need_imgs > unique_labels.shape[0]:
            raise ValueError(
                f"Need train_images+val_images={need_imgs} but only {unique_labels.shape[0]} unique labels exist."
            )
        train_labels = unique_labels[: int(cfg.trial_train_images)]
        val_labels = unique_labels[int(cfg.trial_train_images) : need_imgs]
        train_idx = _indices_from_labels(triplets, train_labels)
        val_idx = _indices_from_labels(triplets, val_labels)

    Zs_train, Zu_train = Zs[train_idx], Zu[train_idx]
    Zs_val, Zu_val = Zs[val_idx], Zu[val_idx]
    train_paired = FmriPairedDataset(sup_name, Zs_train, unsup_name, Zu_train)
    val_paired = FmriPairedDataset(sup_name, Zs_val, unsup_name, Zu_val)
    return train_paired, val_paired, embed_dim


def make_single_view_datasets(
    paired: FmriPairedDataset,
    sup_name: str,
    unsup_name: str,
) -> Tuple[FmriSingleViewDataset, FmriSingleViewDataset]:
    """Convert a paired dataset into two aligned single-view datasets."""
    if paired.name_a == sup_name and paired.name_b == unsup_name:
        return (
            FmriSingleViewDataset(sup_name, paired.Za),
            FmriSingleViewDataset(unsup_name, paired.Zb),
        )
    if paired.name_a == unsup_name and paired.name_b == sup_name:
        return (
            FmriSingleViewDataset(sup_name, paired.Zb),
            FmriSingleViewDataset(unsup_name, paired.Za),
        )
    raise ValueError("Paired dataset names do not match requested view names.")


def load_fmri_train_and_eval_datasets(
    *,
    cfg: FmriSplitConfig,
    sup_name: str,
    unsup_name: str,
    sup_subject: int = 1,
    unsup_subject: int = 2,
) -> Tuple[FmriSingleViewDataset, FmriSingleViewDataset, FmriPairedDataset, int]:
    """Load fMRI train datasets and overlap eval dataset.

    Behavior:
    - train_on_overlap=False (default): train on non-overlap (`Z_train`), eval on overlap (`Z_test`).
    - train_on_overlap=True: train/eval both come from overlap (`Z_test`) using disjoint splits.
    """
    data_dir = _resolve_data_dir(cfg.data_dir)
    if cfg.fmt == "avg":
        template = "avg_ws_mlp_v1_128_768_sub-{subject:02d}.pt"
    elif cfg.fmt == "trial":
        template = "trial_ws_mlp_v1_128_768_sub-{subject:02d}.pt"
    else:
        raise ValueError(f"Unknown fmt: {cfg.fmt}")

    sup_ckpt = _load_subject_checkpoint(data_dir, template, sup_subject)
    unsup_ckpt = _load_subject_checkpoint(data_dir, template, unsup_subject)

    Zs_train_nonoverlap = sup_ckpt["Z_train"].to(torch.float32)
    Zu_train_nonoverlap = unsup_ckpt["Z_train"].to(torch.float32)
    Zs_overlap = sup_ckpt["Z_test"].to(torch.float32)
    Zu_overlap = unsup_ckpt["Z_test"].to(torch.float32)
    if Zs_overlap.shape != Zu_overlap.shape:
        raise ValueError(f"Subject test embeddings mismatch: {Zs_overlap.shape} vs {Zu_overlap.shape}")
    embed_dim = int(Zs_overlap.shape[1])
    if not torch.equal(sup_ckpt["labels_test"], unsup_ckpt["labels_test"]):
        raise ValueError("labels_test must match across subjects for overlap eval.")

    if cfg.fmt == "avg":
        n_overlap = int(Zs_overlap.shape[0])
        perm = _randperm(n_overlap, cfg.seed).numpy()

        if cfg.train_on_overlap:
            need = int(cfg.avg_train_size) + int(cfg.avg_val_size)
            if need > n_overlap:
                raise ValueError(f"Need overlap train+val={need} but only n={n_overlap} exist.")
            train_idx = perm[: int(cfg.avg_train_size)]
            eval_idx = perm[int(cfg.avg_train_size) : need]
            Zs_train = Zs_overlap[train_idx]
            Zu_train = Zu_overlap[train_idx]
        else:
            Zs_train = Zs_train_nonoverlap
            Zu_train = Zu_train_nonoverlap
            eval_n = int(cfg.avg_eval_size)
            if eval_n <= 0 or eval_n > n_overlap:
                eval_n = n_overlap
            eval_idx = perm[:eval_n]

        eval_paired = FmriPairedDataset(
            sup_name,
            Zs_overlap[eval_idx],
            unsup_name,
            Zu_overlap[eval_idx],
        )
    else:
        reps_s = sup_ckpt.get("repetitions_test", None)
        reps_u = unsup_ckpt.get("repetitions_test", None)
        if reps_s is None or reps_u is None:
            raise ValueError("trial format requires repetitions_test in both checkpoints.")
        if not torch.equal(reps_s, reps_u):
            raise ValueError("repetitions_test must match across subjects for overlap eval.")

        triplets = _trial_triplets(sup_ckpt["labels_test"], reps_s)
        unique_labels = np.asarray(sorted(triplets.keys()), dtype=np.int64)
        rng = np.random.default_rng(int(cfg.seed))
        rng.shuffle(unique_labels)

        if cfg.train_on_overlap:
            need_imgs = int(cfg.trial_train_images) + int(cfg.trial_val_images)
            if need_imgs > unique_labels.shape[0]:
                raise ValueError(
                    f"Need overlap train_images+val_images={need_imgs} but only {unique_labels.shape[0]} labels exist."
                )
            train_labels = unique_labels[: int(cfg.trial_train_images)]
            eval_labels = unique_labels[int(cfg.trial_train_images) : need_imgs]
            train_idx = _indices_from_labels(triplets, train_labels)
            Zs_train = Zs_overlap[train_idx]
            Zu_train = Zu_overlap[train_idx]
        else:
            Zs_train = Zs_train_nonoverlap
            Zu_train = Zu_train_nonoverlap
            eval_imgs = int(cfg.trial_eval_images)
            if eval_imgs <= 0 or eval_imgs > unique_labels.shape[0]:
                eval_imgs = unique_labels.shape[0]
            eval_labels = unique_labels[:eval_imgs]

        eval_idx = _indices_from_labels(triplets, eval_labels)
        eval_paired = FmriPairedDataset(
            sup_name,
            Zs_overlap[eval_idx],
            unsup_name,
            Zu_overlap[eval_idx],
        )

    supset = FmriSingleViewDataset(sup_name, Zs_train)
    unsupset = FmriSingleViewDataset(unsup_name, Zu_train)
    return supset, unsupset, eval_paired, embed_dim

