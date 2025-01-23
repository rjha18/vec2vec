from sys import argv
from types import SimpleNamespace
import toml
import torch

from utils.model_utils import load_encoder
from utils.utils import load_n_translator, read_args


cfg = toml.load(f'configs/{argv[1]}.toml')
unknown_cfg = read_args(argv)
cfg = SimpleNamespace(**{**cfg['general'], **cfg['train'], **cfg['logging'], **cfg['huggingface'], **unknown_cfg})

encoders = {emb: load_encoder(emb) for emb in cfg.embs}
encoder_dims = {emb: encoders[emb].get_sentence_embedding_dimension() for emb in cfg.embs}
save_dir = cfg.save_dir.format(cfg.latent_dims if hasattr(cfg, 'latent_dims') else cfg.wandb_name)
translator = load_n_translator(cfg, encoder_dims)
translator.load_state_dict(torch.load(save_dir + f'model_{cfg.load_from_epoch}.pt'))

# save locally
translator.save_pretrained(f'{cfg.model_name}')

# push to the hub
translator.push_to_hub(f'{cfg.hf_username}/{cfg.model_name}')
