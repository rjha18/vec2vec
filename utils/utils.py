import multiprocessing
import os, json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub.file_download import hf_hub_download
from translators import TransformTranslator
from safetensors.torch import load_file

from utils.embeddings import load_and_process_embeddings_from_idxs

from translators.MLPWithResidual import MLPWithResidual
from translators.LinearTranslator import LinearTranslator
from translators.TransformTranslator import TransformTranslator
from translators.transforms.UNetTransform import UNetTransform
from translators.transforms.LMTransform import LMTransform
from translators.transforms.UNet1dTransform import UNet1dTransform

from vec2text.models import InversionModel


def load_n_translator(cfg, encoder_dims):
    if cfg.style == 'linear':
        return LinearTranslator(
            encoder_dims,
            cfg.normalize_embeddings,
            cfg.src_emb if hasattr(cfg, 'src_emb') else None,
            cfg.tgt_emb if hasattr(cfg, 'tgt_emb') else None
        )

    if cfg.style == 'n_simple':
        transform = nn.Linear(cfg.d_adapter, cfg.d_adapter)
    elif cfg.style == 'n_double':
        transform = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_adapter, cfg.d_adapter),
            nn.SiLU(),
            nn.Linear(cfg.d_adapter, cfg.d_adapter),
            nn.SiLU(),
        )
    elif cfg.style == 'res_mlp':
        transform = MLPWithResidual(cfg.transform_depth, cfg.d_adapter, cfg.d_transform, cfg.d_adapter, cfg.norm_style)
    elif cfg.style == 'n_ae':
        transform = nn.Sequential(
            nn.Linear(cfg.d_adapter, cfg.latent_dims),
            nn.ReLU(),
            nn.Linear(cfg.latent_dims, cfg.d_adapter)
        )
    elif cfg.style == 'n_bert':
        transform = LMTransform(cfg.d_adapter, cfg.d_adapter, cfg.lm_base_name, cfg.upscale_num)
    elif cfg.style == 'unet':
        transform = UNetTransform(cfg.d_adapter, cfg.d_adapter)
    elif cfg.style == 'unet1d':
        transform = UNet1dTransform(cfg.d_adapter, cfg.d_adapter)
    else:
        raise ValueError(f"Unknown style: {cfg.style}")

    return TransformTranslator(
        encoder_dims=encoder_dims,
        d_adapter=cfg.d_adapter,
        d_hidden=cfg.d_hidden,
        transform=transform,
        depth=cfg.depth,
        style=cfg.style,
        use_target_vectors=cfg.use_target_vectors,
        use_small_output_adapters=cfg.use_small_output_adapters if hasattr(cfg, 'use_small_output_adapters') else False,
        use_residual_adapters=cfg.use_residual_adapters if hasattr(cfg, 'use_residual_adapters') else False,
        norm_style=cfg.norm_style if hasattr(cfg, 'norm_style') else 'batch',
    )


def get_loaders(dsets, bs, shuffle):
    return [DataLoader(dset, batch_size=bs, shuffle=shuffle, num_workers=1) for dset in dsets]

def get_val_sets(masks, cfg, seed, test_flag=True, keep_in_memory=True):
    mask = masks[0] if not test_flag else ~masks[0]
    for m in masks[1:]:
        mask &= m if not test_flag else ~m

    np.random.seed(seed)
    idxs = np.random.choice(np.where(mask)[0], cfg.val_size, replace=False)
    X_val = load_and_process_embeddings_from_idxs(
        cfg.dataset, cfg.emb1, idxs, cfg.normalize_embeddings, 'train', 32, keep_in_memory, 'cpu'
    )

    Y_val = load_and_process_embeddings_from_idxs(
        cfg.dataset, cfg.emb2, idxs, cfg.normalize_embeddings, 'train', 32, keep_in_memory, 'cpu'
    )
    return X_val, Y_val

def get_text_sets(dsets, size, seed):
    np.random.seed(seed)
    idxs = np.random.choice(len(dsets[0]), size, replace=False)
    return [d.select(idxs) for d in dsets]

def get_text_loader(size):
    return DataLoader(TensorDataset(torch.arange(size)), batch_size=64, shuffle=False)

def get_inverters(emb_flags, device='cpu'):
    assert isinstance(emb_flags, list)
    inverters = {}
    for emb_flag in emb_flags:
        assert emb_flag in ['gtr', 'gte']
        if emb_flag == "gtr":
            # inversion_model = InversionModel.from_pretrained("jxm/gtr-32-noise-0.0001")
            # inversion_model = InversionModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_corrector")
            inversion_model = InversionModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_inversion")
            inversion_model.eval()

        elif emb_flag == 'gte':
            inversion_model = InversionModel.from_pretrained("jxm/gte-32-noise-0.0001")
        inverters[emb_flag] = inversion_model.to(device)
    return inverters

def read_args(argv):
    cfg = {}
    # Handle unknown arguments
    for arg in argv:
        if arg.startswith("--"):
            key = arg.lstrip("--")
            # Attempt to parse value as int, float, or leave as string
            try:
                value = int(argv[argv.index(arg) + 1])
            except ValueError:
                try:
                    value = float(argv[argv.index(arg) + 1])
                except ValueError:
                    value = argv[argv.index(arg) + 1]
            cfg[key] = value
    return cfg


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1

def get_num_proc() -> int:
    world_size: int = torch.cuda.device_count()
    try:
        # os.sched_getaffinity respects schedulers, unlike cpu_count(), but it's only available
        # on some Unix platforms, so we support both!
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size
    

def load_translator_from_hf(model_id):
    if os.path.isdir(model_id):
        print("Loading weights from local directory")
        model_file = os.path.join(model_id, 'model.safetensors')
        config_file = os.path.join(model_id, 'config.json')
    else:
        model_file = hf_hub_download(
            repo_id=model_id,
            filename='model.safetensors',
        )
        config_file = hf_hub_download(
            repo_id=model_id,
            filename='config.json',
        )
    state_dict = load_file(model_file)
    with open(config_file) as f:
        cfg = json.load(f)
    cfg = SimpleNamespace(**cfg)
    translator = load_n_translator(cfg, cfg.encoder_dims)
    translator.load_state_dict(state_dict, strict=False)
    return translator