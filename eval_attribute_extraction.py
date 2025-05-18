import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import pandas as pd
import torch

# from eval import eval_model
from utils.collate import MultiEncoderClassificationDataset, TokenizedCollator
from utils.dist import get_rank
from utils.eval_utils import eval_loop_, text_to_embedding
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings

from datasets import load_dataset, load_from_disk


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
    # https://github.com/huggingface/transformers/issues/26548
    accelerator.dataloader_config.dispatch_batches = False

    sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision != 'no' else None)}
    encoder_dims = {cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])}
    translator = load_n_translator(cfg, encoder_dims)

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {
        cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision != 'no' else None)
    }
    unsup_dim = {
        cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb])
    }
    translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])

    if cfg.style != 'identity':
        assert cfg.unsup_emb not in sup_encs
        assert cfg.unsup_emb in translator.in_adapters
        assert cfg.unsup_emb in translator.out_adapters

        cfg.num_params = sum(x.numel() for x in translator.parameters())
        print("Number of parameters:", cfg.num_params)

    if hasattr(cfg, 'val_dataset') and cfg.val_dataset == 'tweets':
        ### Tweets
        dset_name = 'cardiffnlp/tweet_topic_multilingual'
        dset = load_dataset('cardiffnlp/tweet_topic_multilingual', 'en', num_proc=8)['test']
        raw_labels = pd.read_csv('labels/tweet_topic_multilingual.csv')['label'].tolist()
    elif hasattr(cfg, 'val_dataset') and cfg.val_dataset == 'enron':
        ### ENRON
        dset_name = 'rishi-jha/filtered_enron'
        dset = load_dataset('rishi-jha/filtered_enron', split='train', num_proc=8).shuffle(seed=cfg.val_dataset_seed).select(range(1280))
        raw_labels = list(json.load(open('email_to_index.json', 'r')).keys())
        # read from email_structure.txt
        email_structure = open('email_structure.txt', 'r').read()
        print(email_structure)
        raw_labels = [email_structure.format(l, l) for l in raw_labels]
    elif hasattr(cfg, 'val_dataset') and 'mimic' in cfg.val_dataset:
        ### MIMIC
        if cfg.val_dataset == 'mimic':
            dset_name = 'data/mimic'
        elif cfg.val_dataset == 'mimic_templates':
            dset_name = 'data/mimic_templates'
        split = "medcat"
        dset = load_from_disk(dset_name)['unsupervised'].shuffle(seed=cfg.val_dataset_seed)
        dset = dset.select(range(cfg.val_size))
        raw_labels = pd.read_csv(f"data/mimic/{split}_mapping.csv").sort_values("index")[split + '_description' if split == 'medcat' else ''].to_list()
        num_classes = len(raw_labels)

        def add_one_hot_label(example):
            index = example[f"{split}" + "_indices" if split == 'medcat' else "_index"]
            one_hot = [0] * num_classes

            if isinstance(index, list):
                for i in index:
                    one_hot[i] = 1
            elif isinstance(index, int):
                one_hot[index] = 1
            else:
                raise ValueError(f"Unknown index type {type(index)}")

            example["label"] = one_hot
            return example

        dset = dset.map(add_one_hot_label)
        keep_columns = ["text", "label"]
        dset = dset.remove_columns([col for col in dset.column_names if col not in keep_columns])
    else:
        raise ValueError(f"Unknown dataset {cfg.val_dataset}")


    # Labels for attribute extraction
    labels = {
        cfg.sup_emb: text_to_embedding(raw_labels, cfg.sup_emb, sup_encs[cfg.sup_emb], cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device),
        cfg.unsup_emb: text_to_embedding(raw_labels, cfg.unsup_emb, unsup_enc[cfg.unsup_emb], cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device)
    }
    print('Loaded labels...')
    num_workers = get_num_proc()

    dset = MultiEncoderClassificationDataset(
        dataset=dset,
        encoders={ **unsup_enc, **sup_encs },
        n_embs_per_batch=2,
        batch_size=cfg.val_bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )

    valloader = DataLoader(
        dset,
        batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    valloader = accelerator.prepare(valloader)

    if cfg.style != 'identity':
        assert hasattr(cfg, 'load_dir')
        print(f"Loading models from {argv[1]}...")
        translator.load_state_dict(torch.load(f'{argv[1]}/model.pt', map_location='cpu'), strict=False)

    translator = accelerator.prepare(translator)
    inverters = None
    # get_inverters(["gtr"], accelerator.device)

    with torch.no_grad():
        translator.eval()
        val_res = {}
        recons, trans, heatmap_dict, text_recons, text_trans, classification =\
            eval_loop_(
                cfg,
                translator,
                {**sup_encs, **unsup_enc},
                valloader,
                inverters=inverters,
                device=accelerator.device,
                labels=labels
            )
        val_res['recons'] = {}
        for flag, res in recons.items():
            for k, v in res.items():
                if k == 'cos':
                    val_res['recons'][f"rec_{flag}_{k}"] = v

        val_res['trans'] = {}
        for target_flag, d in trans.items():
            for flag, res in d.items():
                for k, v in res.items():
                    if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                        continue
                    val_res['trans'][f"{flag}_{target_flag}_{k}"] = v

        val_res['heatmap'] = {}
        if len(heatmap_dict) > 0:
            for k,v in heatmap_dict.items():
                # if v is a plt.Figure, skip it
                if v.__class__.__name__ == 'Figure':
                    # val_res['heatmap'][f"{k}"] = v
                    continue
                else:
                    val_res['heatmap'][f"{k} (avg. {cfg.top_k_batches} batches)"] = v

        val_res['text_recons'] = {}
        if len(text_recons) > 0:
            for flag, res in text_recons.items():
                for k,v in res.items():
                    val_res['text_recons'][f"text_{k}"] = v

        val_res['text_trans'] = {}
        if len(text_trans) > 0:
            for target_flag, d in text_trans.items():
                for flag, res in d.items():
                    for k, v in res.items():
                        if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                            continue
                        val_res['text_trans'][f"{flag}_{target_flag}_{k}"] = v
        
        val_res['classification'] = {}
        if len(classification) > 0:
            for k,v in classification.items():
                val_res['classification'][f"{k}"] = v


    # write dictionary to file in results including dataset name, embedding names
    if cfg.style == 'identity':
        fnm = f'results/baseline_{dset_name.replace("/", "_")}_{cfg.unsup_emb}_{cfg.sup_emb}.json'
    else:
        fnm = f'results/{dset_name.replace("/", "_")}_{cfg.unsup_emb}_{cfg.sup_emb}.json'
    with open(fnm, 'w') as f:
        # human readable
        f.write(json.dumps(val_res, indent=4))
    # save results to file



if __name__ == "__main__":
    main()

