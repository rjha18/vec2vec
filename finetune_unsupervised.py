import itertools
import math
import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate
from tqdm import tqdm
import wandb

import torch
from torch.optim.lr_scheduler import LambdaLR

from eval import eval_model
from utils.collate import MultiEncoderCollator
from utils.dist import get_rank, get_world_size
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.eval_utils import EarlyStopper, eval_loop_
from utils.utils import *
from utils.train_utils import uni_loss_fn, vsp_loss_fn, get_grad_norm
from utils.streaming_utils import load_streaming_embeddings, process_batch
from utils.wandb_logger import Logger


def training_loop_(
    save_dir, accelerator, translator, sup_dataloader, unsup_dataloader, sup_encs, unsup_enc, cfg, opt, scheduler, logger=None, max_num_batches=None
):
    device = accelerator.device
    if logger is None:
        logger = Logger(dummy=True)

    if get_rank() == 0:
        try:
            wandb.watch(translator, log='all')
        except:
            pass

    if get_rank() == 0:
        dataloader_pbar = tqdm(zip(sup_dataloader, unsup_dataloader), total=len(sup_dataloader), desc="Training")
    else:
        dataloader_pbar = zip(sup_dataloader, unsup_dataloader)

    model_save_dir = os.path.join(save_dir, 'model.pt')
    for i, (sup_batch, unsup_batch) in enumerate(dataloader_pbar):
        if max_num_batches is not None and i >= max_num_batches:
            print(f"Early stopping at {i} batches")
            break
        with accelerator.accumulate(translator), accelerator.autocast():
            # assert that there is no overlap in keys (unsupervised assumption)
            assert len(set(sup_batch.keys()).intersection(unsup_batch.keys())) == 0

            ins = {**process_batch(cfg, sup_batch, sup_encs, device), **process_batch(cfg, unsup_batch, unsup_enc, device)}

            if cfg.add_noise:
                min_noise_pow = -10
                max_noise_pow = -1
            else:
                min_noise_pow = 0
                max_noise_pow = 0

            recons, translations = (
                accelerator.unwrap_model(translator).forward(ins, max_noise_pow, min_noise_pow, in_set={cfg.unsupervised_emb})
            )

            if cfg.loss_coefficient_cc_back > 0:
                keys = set(ins.keys())
                keys.remove(cfg.unsupervised_emb)
                _, back_translations = (
                    accelerator.unwrap_model(translator).forward(ins, max_noise_pow, min_noise_pow, in_set=keys, out_set={cfg.unsupervised_emb})
                )

            # recons should have 1 key: unsupervised_emb
            assert len(recons) == 1 and cfg.unsupervised_emb in recons
            rec_loss = uni_loss_fn(ins[cfg.unsupervised_emb], recons[cfg.unsupervised_emb], cfg.unsupervised_emb, cfg.unsupervised_emb, logger)

            if cfg.loss_coefficient_vsp > 0:
                vsp_loss = vsp_loss_fn(ins, translations, logger)
            else:
                vsp_loss = torch.tensor(0.0)

            if cfg.loss_coefficient_cc > 0:
                cc_keys = list(translations.keys())
                random.shuffle(cc_keys)
                cc_translations = dict(
                    itertools.chain(*[{ k1: v.detach() for v in translations[k1].values()}.items() for k1 in cc_keys]))
                    # itertools.chain(*[{ k1: v for v in translations[k1].values()}.items() for k1 in cc_keys]))
                _, cc_translations = (
                    accelerator.unwrap_model(translator).forward(cc_translations, max_noise_pow, min_noise_pow, out_set={cfg.unsupervised_emb})
                )
                cc_loss = torch.tensor(0.0).to(device)
                for flag, emb in cc_translations[cfg.unsupervised_emb].items():
                    cc_loss += uni_loss_fn(ins[cfg.unsupervised_emb], emb, cfg.unsupervised_emb, f'via_{flag}', logger)
            else:
                cc_loss = torch.tensor(0.0)

            if cfg.loss_coefficient_cc_back > 0:
                back_cc_loss = torch.tensor(0.0).to(device)
                for target_flag, emb in back_translations[cfg.unsupervised_emb].items():
                    trans = accelerator.unwrap_model(translator).translate_embeddings(emb.detach(), cfg.unsupervised_emb, target_flag, max_noise_pow, min_noise_pow)
                    back_cc_loss += uni_loss_fn(ins[target_flag], trans, target_flag, f'via_{cfg.unsupervised_emb}', logger)
            else:
                back_cc_loss = torch.tensor(0.0)

            loss = (
                  (rec_loss * cfg.loss_coefficient_rec)
                + (vsp_loss * cfg.loss_coefficient_vsp)
                + (cc_loss * cfg.loss_coefficient_cc)
                + (back_cc_loss * cfg.loss_coefficient_cc_back)
            )

            opt.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(translator.parameters(), cfg.max_grad_norm)
            grad_norm = get_grad_norm(translator)

            opt.step()
            scheduler.step()

            metrics = {
                "rec_loss": rec_loss.item(),
                "vsp_loss": vsp_loss.item(),
                "cc_loss": cc_loss.item(),
                "back_cc_loss": back_cc_loss.item(),
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "learning_rate": opt.param_groups[0]["lr"],
            }

            for metric, value in metrics.items():
                logger.logkv(metric, value)
            logger.dumpkvs(force=(hasattr(cfg, 'force_dump') and cfg.force_dump))
            if get_rank() == 0:
                dataloader_pbar.set_postfix(metrics)

        if (i + 1) % cfg.save_every == 0:
            # save config
            with open(save_dir + 'config.toml', 'w') as f:
                toml.dump(cfg.__dict__, f)
            # save model
            torch.save(accelerator.unwrap_model(translator).state_dict(), model_save_dir)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'configs/{argv[1]}.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**cfg['general'], **cfg['train'], **cfg['logging'], **unknown_cfg})

    use_val_set = hasattr(cfg, 'val_size')

    assert hasattr(cfg, 'load_dir')
    assert hasattr(cfg, 'unsupervised_emb')

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )
    # https://github.com/huggingface/transformers/issues/26548
    accelerator.dataloader_config.dispatch_batches = False

    if hasattr(cfg, 'force_wandb_name') and cfg.force_wandb_name:
        save_dir = cfg.save_dir.format(cfg.wandb_name)
    else:
        cfg.wandb_name = ','.join([f"{k[0]}:{v}" for k, v in unknown_cfg.items()]) if unknown_cfg else cfg.wandb_name
        save_dir = cfg.save_dir.format(cfg.latent_dims if hasattr(cfg, 'latent_dims') else cfg.wandb_name)

    print("Running Experiment:", cfg.wandb_name)

    dset = load_streaming_embeddings(cfg.dataset)

    sup_encs = {emb: load_encoder(emb, mixed_precision='bf16' == cfg.mixed_precision) for emb in cfg.embs}
    encoder_dims = {emb: get_sentence_embedding_dimension(sup_encs[emb]) for emb in cfg.embs}
    translator = load_n_translator(cfg, encoder_dims)

    model_save_dir = os.path.join(save_dir, 'model.pt')

    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading model from {cfg.load_dir}...")
    translator.load_state_dict(torch.load(cfg.load_dir, map_location='cpu'), strict=False)

    if hasattr(cfg, 'freeze_params') and cfg.freeze_params:
        for param in translator.parameters():
            param.requires_grad = False
    
    if cfg.unsupervised_emb not in cfg.embs:
        unsup_enc = {cfg.unsupervised_emb: load_encoder(cfg.unsupervised_emb, mixed_precision='bf16' == cfg.mixed_precision)}
    else:
        unsup_enc = {cfg.unsupervised_emb: sup_encs.pop(cfg.unsupervised_emb)}
        cfg.embs.remove(cfg.unsupervised_emb)

    unsup_dim = {cfg.unsupervised_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsupervised_emb])}
    translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsupervised_emb])

    assert cfg.unsupervised_emb not in cfg.embs
    assert cfg.unsupervised_emb not in sup_encs
    assert cfg.unsupervised_emb in translator.in_adapters
    assert cfg.unsupervised_emb in translator.out_adapters

    cfg.num_params = sum(x.numel() for x in translator.parameters())
    print("Number of parameters:", cfg.num_params)
    print("Number of *trainable* parameters:", sum(p.numel() for p in translator.parameters() if p.requires_grad))
    print("Number of training datapoints:", len(dset))

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=(cfg.wandb_project is None) or not (cfg.use_wandb),
        config=cfg,
    )

    random.seed(cfg.seed + get_rank())
    torch.manual_seed(cfg.seed + get_rank())

    num_workers = get_num_proc()
    print(f"Rank {get_rank()} using {num_workers} workers and {len(dset)} datapoints")
    if get_world_size() > 1:
        max_num_datapoints = len(dset) - (len(dset) % (cfg.bs * get_world_size()))
        dset = dset.select(range(max_num_datapoints))
        print(f"[Filtered] Rank {get_rank()} now using {len(dset)} datapoints")

    mask = np.full(len(dset), False)
    if hasattr(cfg, 'num_points'):
        mask[:cfg.num_points] = True
        np.random.seed(cfg.seed + get_rank())
        np.random.shuffle(mask)
        supset = dset.select(np.where(mask)[0])
        unsupset = dset.select(np.where(~mask)[0])
        unsupmask = np.full(len(unsupset), False)
        unsupmask[:cfg.num_points] = True
        np.random.seed(cfg.seed + get_rank())
        np.random.shuffle(unsupmask)
        if use_val_set:
            valset = unsupset.select(np.where(~unsupmask)[0])
            valmask = np.full(len(valset), False)
            valmask[:cfg.val_size] = True
            np.random.seed(cfg.seed + get_rank())
            np.random.shuffle(valmask)
            valset = valset.select(np.where(valmask)[0])
        unsupset = unsupset.select(np.where(unsupmask)[0])

    sup_dataloader = DataLoader(
        supset, 
        batch_size=cfg.bs, 
        num_workers=num_workers, 
        shuffle=True, 
        pin_memory=True, 
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=MultiEncoderCollator(
            sup_encs, cfg.n_embs_per_batch, max_length=cfg.max_seq_length
        ),
        drop_last=True,
    )

    unsup_dataloader = DataLoader(
        unsupset,
        batch_size=cfg.bs,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=MultiEncoderCollator(
            unsup_enc, 1, max_length=cfg.max_seq_length
        ),
        drop_last=True,
    )

    if use_val_set:
        valloader = DataLoader(
            valset, 
            batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
            num_workers=num_workers, 
            shuffle=False, 
            pin_memory=True, 
            prefetch_factor=(8 if num_workers > 0 else None),
            collate_fn=MultiEncoderCollator(
                {**sup_encs, **unsup_enc}, len({**sup_encs, **unsup_enc}), max_length=cfg.max_seq_length
            ),
            drop_last=True,
        )
        valloader = accelerator.prepare(valloader)

    max_num_epochs = int(math.ceil(cfg.epochs))

    opt = torch.optim.AdamW(translator.parameters(), lr=cfg.lr, fused=False, weight_decay=0.01)
    steps_per_epoch = len(supset) // cfg.bs
    total_steps = steps_per_epoch * cfg.epochs / cfg.gradient_accumulation_steps
    scheduler = LambdaLR(opt, lr_lambda=lambda step: 1 - step / max(1, total_steps))
    translator, opt, scheduler, sup_dataloader, unsup_dataloader = accelerator.prepare(
        translator, opt, scheduler, sup_dataloader, unsup_dataloader
    )


    best_model = None
    early_stopper = EarlyStopper(
        patience=cfg.patience if hasattr(cfg, 'patience') else 1,
        min_delta=cfg.delta if hasattr(cfg, 'delta') else 0
    )

    for epoch in range(max_num_epochs):
        stop_cond = []
        if use_val_set and get_rank() == 0:
            with torch.no_grad(), accelerator.autocast():
                translator.eval()
                val_res = {}
                recons, trans = eval_loop_(cfg, translator, {**sup_encs, **unsup_enc}, valloader, device=accelerator.device)
                for flag, res in recons.items():
                    for k, v in res.items():
                        if k == 'cos':
                            val_res[f"val/rec_{flag}_{k}"] = v
                for target_flag, d in trans.items():
                    for flag, res in d.items():
                        for k, v in res.items():
                            if k == 'cos':
                                if flag == 'dpr' and target_flag == 'dpr':
                                    continue
                                elif flag == 'dpr':
                                    stop_cond.append(v)
                                    val_res[f"val_from/{flag}_{target_flag}_{k}"] = v
                                elif target_flag == 'dpr':
                                    val_res[f"val_to/{flag}_{target_flag}_{k}"] = v
                                else:
                                    val_res[f"val_{flag}_{target_flag}_{k}"] = v

                val_res['stop_cond'] = sum(stop_cond) / len(stop_cond)
                wandb.log(val_res)

                if early_stopper.early_stop(val_res['stop_cond']):
                    print("Stopping early! Saving previous model...")
                    break
                else:
                    best_model = accelerator.unwrap_model(translator).state_dict().copy()

        max_num_batches = None
        print(f"Epoch", epoch, "max_num_batches", max_num_batches, "max_num_epochs", max_num_epochs)
        if epoch + 1 >= max_num_epochs:
            max_num_batches = max(1, (cfg.epochs - epoch) * len(supset) // cfg.bs // get_world_size())
            print(f"Setting max_num_batches to {max_num_batches}")

        translator.train()
        training_loop_(
            save_dir=save_dir,
            accelerator=accelerator,
            translator=translator,
            sup_dataloader=sup_dataloader,
            unsup_dataloader=unsup_dataloader,
            sup_encs=sup_encs,
            unsup_enc=unsup_enc,
            cfg=cfg,
            opt=opt,
            scheduler=scheduler,
            logger=logger,
            max_num_batches=max_num_batches
        )

    with open(save_dir + 'config.toml', 'w') as f:
        toml.dump(cfg.__dict__, f)
    # save model
    best_model = best_model if best_model is not None else accelerator.unwrap_model(translator).state_dict()
    torch.save(best_model, model_save_dir)

    # eval
    cfg.use_good_queries = 1
    cfg.dataset = "NanoNQ"
    cfg.max_seq_length = cfg.max_seq_length
    cfg.batch_size = cfg.bs
    cfg.train_path = save_dir
    metrics = eval_model(cfg=cfg, translator=translator)
    wandb.log({ f"eval/k": v for k,v in metrics.items() })

if __name__ == "__main__":
    main()
