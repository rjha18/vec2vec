"""Encode a text dataset with one or more encoders; save paired .pt tensors.

Rows stay aligned across encoders (same texts, same order), so the outputs
are ground-truth paired embeddings for cross-dataset eval.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from utils.model_utils import load_encoder
from utils.streaming_utils import load_streaming_embeddings


def load_texts(dataset_name, n, seed):
    """Stream large web corpora (avoid loading the whole sample into RAM -> OOM)."""
    if dataset_name in {"fineweb", "fineweb-medium", "fineweb-tiny"}:
        cfg = {"fineweb": ("HuggingFaceFW/fineweb", None),
               "fineweb-medium": ("HuggingFaceFW/fineweb", "sample-350BT"),
               "fineweb-tiny": ("HuggingFaceFW/fineweb", "sample-10BT")}[dataset_name]
        repo, name = cfg
        dset = load_dataset(repo, name, split="train", streaming=True) if name \
            else load_dataset(repo, split="train", streaming=True)
        rows = [r["text"] for r in dset.shuffle(seed=seed, buffer_size=max(1000, min(10000, n * 4))).take(n)]
        return rows
    dset = load_streaming_embeddings(dataset_name)
    dset = dset.shuffle(seed=seed).select(range(min(n, len(dset))))
    col = "text" if "text" in dset.column_names else dset.column_names[0]
    return [str(t) for t in dset[col]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--encoders", required=True, help="comma-separated flags, e.g. gte,gtr")
    p.add_argument("--n", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_length", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--raw", action="store_true", help="keep raw norms (skip unit-normalization) — norm is a rotation-invariant per-point signal")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = load_texts(args.dataset, args.n, args.seed)
    print(f"{len(texts)} texts from {args.dataset}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name in args.encoders.split(","):
        enc = load_encoder(name, device=device)
        if hasattr(enc, "max_seq_length"):
            enc.max_seq_length = args.max_seq_length
        emb = enc.encode(texts, batch_size=args.batch_size, convert_to_tensor=True,
                         normalize_embeddings=not args.raw, show_progress_bar=False, device=device).float().cpu()
        safe = name.replace("/", "_")
        torch.save(emb, out / f"{safe}.pt")
        print(f"saved {name}: {tuple(emb.shape)} -> {out / f'{safe}.pt'}")
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
