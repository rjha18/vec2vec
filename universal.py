import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE  # Use TSNE instead of PCA

# from eval import eval_model
from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.dist import get_rank
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings, process_batch


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'{argv[1]}/config.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**cfg, **unknown_cfg})

    if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        cfg.mixed_precision = 'fp16'
        print("Note: bf16 is not available on this hardware!")

    # set seeds
    random.seed(cfg.seed + get_rank())
    torch.manual_seed(cfg.seed + get_rank())
    np.random.seed(cfg.seed + get_rank())
    torch.cuda.manual_seed(cfg.seed + get_rank())

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None
    )
    accelerator.dataloader_config.dispatch_batches = False

    dset = load_streaming_embeddings(cfg.dataset)

    sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)}
    encoder_dims = {cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])}
    translator = load_n_translator(cfg, encoder_dims)

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {
        cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)
    }
    unsup_dim = {
        cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb])
    }
    translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])

    assert cfg.unsup_emb not in sup_encs
    assert cfg.unsup_emb in translator.in_adapters
    assert cfg.unsup_emb in translator.out_adapters

    cfg.num_params = sum(x.numel() for x in translator.parameters())
    print("Number of parameters:", cfg.num_params)

    dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    dset = dset_dict["train"]
    valset = dset_dict["test"]

    assert hasattr(cfg, 'num_points') or hasattr(cfg, 'unsup_points')
    dset = dset.shuffle(seed=cfg.train_dataset_seed)
    if hasattr(cfg, 'num_points'):
        assert cfg.num_points > 0 and cfg.num_points <= len(dset) // 2
        unsupset = dset.select(range(cfg.num_points, cfg.num_points + cfg.val_size))
    elif hasattr(cfg, 'unsup_points'):
        unsupset = dset.select(range(min(cfg.unsup_points, len(cfg.val_size))))

    num_workers = get_num_proc()
    evalset = MultiencoderTokenizedDataset(
        dataset=valset if hasattr(cfg, 'use_ood') and cfg.use_ood else unsupset,
        encoders={**unsup_enc, **sup_encs},
        n_embs_per_batch=2,
        batch_size=cfg.val_bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )
    evalloader = DataLoader(
        evalset,
        batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    evalloader = accelerator.prepare(evalloader)

    assert hasattr(cfg, 'load_dir')
    print(f"Loading models from {argv[1]}...")
    translator.load_state_dict(torch.load(f'{argv[1]}/model.pt', map_location='cpu'), strict=False)
    translator = accelerator.prepare(translator)


    ALPHA = 0.7
    WIDTH = 0.5
    C1 = '#008080'
    C2 = '#FF6347'
    LINEWIDTH = 0.5
    with torch.no_grad():
        translator.eval()
        batch = next(iter(evalloader))

        ins = process_batch(batch, {**sup_encs, **unsup_enc}, cfg.normalize_embeddings, accelerator.device)
        _, _, reps = translator(ins, include_reps=True)
        print(reps[cfg.sup_emb].shape)
        print(reps[cfg.unsup_emb].shape)
        print("Latents", torch.nn.functional.cosine_similarity(reps[cfg.sup_emb], reps[cfg.unsup_emb]).mean())
        print("Inputs", torch.nn.functional.cosine_similarity(ins[cfg.sup_emb], ins[cfg.unsup_emb]).mean())

        # Set up seaborn style and figure
        sns.set_style("white")
        sns.set_context("paper", font_scale=4)
        # Create subplots with more space between them
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.subplots_adjust(wspace=0.5) 

        # First subplot - Input embeddings
        sup_array = ins[cfg.sup_emb].cpu().numpy()
        unsup_array = ins[cfg.unsup_emb].cpu().numpy()
        combined = np.concatenate([sup_array, unsup_array], axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        combined_2d = tsne.fit_transform(combined)
        sup_2d = combined_2d[:sup_array.shape[0]]
        unsup_2d = combined_2d[sup_array.shape[0]:]

        ax1.scatter(sup_2d[:, 0], sup_2d[:, 1], color=C1, label=f'{cfg.sup_emb}', 
                   s=100, edgecolors='black', linewidth=1.0, zorder=2)
        ax1.scatter(unsup_2d[:, 0], unsup_2d[:, 1], color=C2, label=f'{cfg.unsup_emb}', 
                   s=100, edgecolors='black', linewidth=1.0, zorder=2)
        # Draw light lines between paired points
        for i in range(sup_2d.shape[0]):
            ax1.plot([sup_2d[i, 0], unsup_2d[i, 0]],
                     [sup_2d[i, 1], unsup_2d[i, 1]],
                     color='gray', alpha=ALPHA, linewidth=WIDTH, zorder=1)
        # Show spines but hide ticks and labels
        for spine in ax1.spines.values():
            spine.set_linewidth(2.0)
        ax1.spines['top'].set_visible(True)
        ax1.spines['right'].set_visible(True)
        ax1.spines['bottom'].set_visible(True)
        ax1.spines['left'].set_visible(True)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        # Second subplot - Intermediate representations
        sup_array = reps[cfg.sup_emb].cpu().numpy()
        unsup_array = reps[cfg.unsup_emb].cpu().numpy()
        combined = np.concatenate([sup_array, unsup_array], axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        combined_2d = tsne.fit_transform(combined)
        sup_rep_2d = combined_2d[:sup_array.shape[0]]
        unsup_rep_2d = combined_2d[sup_array.shape[0]:]

        ax2.scatter(sup_rep_2d[:, 0], sup_rep_2d[:, 1], color=C1, label=cfg.sup_emb, 
                   s=100, edgecolors='black', linewidth=1.0, zorder=2)
        ax2.scatter(unsup_rep_2d[:, 0], unsup_rep_2d[:, 1], color=C2, label=cfg.unsup_emb, 
                   s=100, edgecolors='black', linewidth=1.0, zorder=2)
        # Draw light lines between paired points
        for i in range(sup_rep_2d.shape[0]):
            ax2.plot([sup_rep_2d[i, 0], unsup_rep_2d[i, 0]],
                     [sup_rep_2d[i, 1], unsup_rep_2d[i, 1]],
                     color='gray', alpha=ALPHA+0.15, linewidth=WIDTH + 0.25, zorder=1)
        # Show spines but hide ticks and labels
        for spine in ax2.spines.values():
            spine.set_linewidth(2.0)
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)
        ax2.spines['bottom'].set_visible(True)
        ax2.spines['left'].set_visible(True)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        # Add arrow between subplots
        # plt.annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.5),
        #             # xycoords='figure fraction', textcoords='figure fraction',
        #             arrowprops=dict(arrowstyle='->', color='black', lw=3),
        #             ha='center', va='center',
        # )

        # Save
        plt.tight_layout()
        plt.savefig(f'results_universal/universal_{cfg.unsup_emb}_{cfg.sup_emb}.png', dpi=300, bbox_inches='tight')
        plt.clf()
        print('Saved plot to universal.png')


if __name__ == "__main__":
    main()