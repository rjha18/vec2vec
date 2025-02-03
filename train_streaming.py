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
from utils.train_utils import rec_loss_fn, trans_loss_fn, uni_loss_fn, vsp_loss_fn, get_grad_norm
from utils.streaming_utils import load_streaming_embeddings, process_batch
from utils.wandb_logger import Logger


def training_loop_(
    accelerator, translator, dataloader, encoders, cfg, opt, scheduler, logger=None, max_num_batches=None
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
        dataloader_pbar = tqdm(dataloader)
    else:
        dataloader_pbar = dataloader

    model_save_dir = os.path.join(cfg.save_dir, 'model.pt')
    for i, batch in enumerate(dataloader_pbar):
        if max_num_batches is not None and i >= max_num_batches:
            print(f"Early stopping at {i} batches")
            break
        with accelerator.accumulate(translator), accelerator.autocast():
            ins = process_batch(cfg, batch, encoders, device)

            if cfg.add_noise:
                min_noise_pow = -10
                max_noise_pow = -1
            else:
                min_noise_pow = 0
                max_noise_pow = 0

            if cfg.targeted:
                trans = accelerator.unwrap_model(translator).translate_embeddings(
                    ins[cfg.src_emb], cfg.src_emb, cfg.tgt_emb, max_noise_pow, min_noise_pow
                )
                rec_loss = torch.tensor(0.0)
                trans_loss = uni_loss_fn(ins[cfg.tgt_emb], trans, cfg.src_emb, cfg.tgt_emb, logger)
            else:
                recons, translations = (
                    accelerator.unwrap_model(translator).forward(ins, max_noise_pow, min_noise_pow)
                )
                rec_loss = rec_loss_fn(ins, recons, logger)
                trans_loss = trans_loss_fn(ins, translations, logger)

            if cfg.loss_coefficient_vsp > 0:
                vsp_loss = vsp_loss_fn(ins, translations, logger)
            else:
                vsp_loss = torch.tensor(0.0)

            if cfg.loss_coefficient_cc > 0:
                cc_keys = list(translations.keys())
                random.shuffle(cc_keys)
                cc_translations = dict(
                    itertools.chain(*[{ k1: v.detach()  for v in translations[k1].values()}.items() for k1 in cc_keys]))
                cc_recons, cc_translations = (
                    accelerator.unwrap_model(translator).forward(cc_translations, max_noise_pow, min_noise_pow)
                )
                cc_rec_loss = rec_loss_fn(ins, cc_recons, logger)
                cc_trans_loss = trans_loss_fn(ins, cc_translations, logger)
            else:
                cc_rec_loss = torch.tensor(0.0)
                cc_trans_loss = torch.tensor(0.0)

            loss = (
                  (rec_loss * cfg.loss_coefficent_rec)
                + (trans_loss * cfg.loss_coefficient_trans)
                + (vsp_loss * cfg.loss_coefficient_vsp)
                + ((cc_rec_loss + cc_trans_loss) * cfg.loss_coefficent_rec)
            )

            opt.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(translator.parameters(), cfg.max_grad_norm)
            grad_norm = get_grad_norm(translator)

            opt.step()
            scheduler.step()

            metrics = {
                "rec_loss": rec_loss.item(),
                "trans_loss": trans_loss.item(),
                "vsp_loss": vsp_loss.item(),
                "cc_rec_loss": cc_rec_loss.item(),
                "cc_trans_loss": cc_trans_loss.item(),
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "learning_rate": opt.param_groups[0]["lr"],
            }

            # print("[7] logging")
            for metric, value in metrics.items():
                logger.logkv(metric, value)
            logger.dumpkvs(force=(hasattr(cfg, 'force_dump') and cfg.force_dump))
            if get_rank() == 0:
                dataloader_pbar.set_postfix(metrics)

        if (i + 1) % cfg.save_every == 0:
            # save config
            with open(cfg.save_dir + 'config.toml', 'w') as f:
                toml.dump(cfg.__dict__, f)
            # save model
            torch.save(accelerator.unwrap_model(translator).state_dict(), model_save_dir)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'configs/{argv[1]}.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**cfg['general'], **cfg['train'], **cfg['logging'], **unknown_cfg})

    cfg.targeted = hasattr(cfg, 'src_emb') and hasattr(cfg, 'tgt_emb')
    use_val_set = hasattr(cfg, 'val_size')
    finetune_mode = hasattr(cfg, 'ft_embs')

    if finetune_mode:
        print("Finetuning mode!")
        assert hasattr(cfg, 'load_dir')

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )
    # https://github.com/huggingface/transformers/issues/26548
    accelerator.dataloader_config.dispatch_batches = False

    if hasattr(cfg, 'force_wandb_name') and cfg.force_wandb_name:
        cfg.save_dir = cfg.save_dir.format(cfg.wandb_name)
    else:
        cfg.wandb_name = ','.join([f"{k[0]}:{v}" for k, v in unknown_cfg.items()]) if unknown_cfg else cfg.wandb_name
        cfg.save_dir = cfg.save_dir.format(cfg.latent_dims if hasattr(cfg, 'latent_dims') else cfg.wandb_name)

    print("Running Experiment:", cfg.wandb_name)

    dset = load_streaming_embeddings(cfg.dataset)

    encoders = {emb: load_encoder(emb, mixed_precision='bf16' == cfg.mixed_precision) for emb in cfg.embs}
    encoder_dims = {emb: get_sentence_embedding_dimension(encoders[emb]) for emb in cfg.embs}
    translator = load_n_translator(cfg, encoder_dims)

    model_save_dir = os.path.join(cfg.save_dir, 'model.pt')
    # model_save_dir = "checkpoints-unet/d:1024,m:bf16,w:11-15-pretrain-full-3,e:5.0,d:fineweb-medium,u:1,b:256,s:44/model.pt"


    os.makedirs(cfg.save_dir, exist_ok=True)
    if hasattr(cfg, 'load_from_epoch'):
        translator.load_state_dict(torch.load(cfg.load_dir + f'model_{cfg.load_from_epoch}.pt'))
    elif finetune_mode and not (hasattr(cfg, 'overwrite') and cfg.overwrite):
        print(f"Loading model from {cfg.load_dir}...")
        translator.load_state_dict(torch.load(cfg.load_dir, map_location='cpu'), strict=False)

    if finetune_mode:
        if hasattr(cfg, 'freeze_params') and cfg.freeze_params:
            for param in translator.parameters():
                param.requires_grad = False
        encoders = {emb: load_encoder(emb, mixed_precision='bf16' == cfg.mixed_precision) for emb in cfg.ft_embs}
        encoder_dims = {emb: get_sentence_embedding_dimension(encoders[emb]) for emb in cfg.ft_embs}
        translator.add_encoders(encoder_dims, overwrite_embs=[cfg.overwrite_emb] if hasattr(cfg, 'overwrite_emb') else None)
        cfg.embs = list(set(cfg.embs + cfg.ft_embs))
        print("Trainable parameters:", sum(p.numel() for p in translator.parameters() if p.requires_grad))

    cfg.num_params = sum(x.numel() for x in translator.parameters())
    print("Number of parameters:", cfg.num_params)
    print("Number of (total) training datapoints:", len(dset))

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
        trainset = dset.select(np.where(mask)[0])
        print("Number of used training datapoints:", len(trainset))

    if use_val_set:
        valset = dset.select(np.where(~mask)[0])
        valmask = np.full(len(valset), False)
        valmask[:cfg.val_size] = True
        np.random.seed(cfg.seed + get_rank())
        np.random.shuffle(valmask)
        valset = valset.select(np.where(valmask)[0])
        print("Number of validation datapoints:", len(trainset))

    dataloader = DataLoader(
        trainset, 
        batch_size=cfg.bs, 
        num_workers=num_workers, 
        shuffle=True, 
        pin_memory=True, 
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=MultiEncoderCollator(
            encoders, cfg.n_embs_per_batch, max_length=cfg.max_seq_length
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
                encoders, cfg.n_embs_per_batch, max_length=cfg.max_seq_length
            ),
            drop_last=True,
        )
        valloader = accelerator.prepare(valloader)

    max_num_epochs = int(math.ceil(cfg.epochs))

    opt = torch.optim.AdamW(translator.parameters(), lr=cfg.lr, fused=False, weight_decay=0.00)
    steps_per_epoch = len(trainset) // cfg.bs
    total_steps = steps_per_epoch * cfg.epochs / cfg.gradient_accumulation_steps
    scheduler = LambdaLR(opt, lr_lambda=lambda step: 1 - step / max(1, total_steps))
    translator, opt, scheduler, dataloader = accelerator.prepare(
        translator, opt, scheduler, dataloader
    )

    if cfg.targeted:
        best_model = None
        early_stopper = EarlyStopper(
            patience=cfg.patience if hasattr(cfg, 'patience') else 1,
            min_delta=cfg.delta if hasattr(cfg, 'delta') else 0
        )
    for epoch in range(max_num_epochs):
        max_num_batches = None
        print(f"Epoch", epoch, "max_num_batches", max_num_batches, "max_num_epochs", max_num_epochs)
        if epoch + 1 >= max_num_epochs:
            max_num_batches = max(1, (cfg.epochs - epoch) * len(trainset) // cfg.bs // get_world_size())
            print(f"Setting max_num_batches to {max_num_batches}")

        translator.train()
        training_loop_(
            accelerator=accelerator,
            translator=translator,
            dataloader=dataloader,
            encoders=encoders,
            cfg=cfg,
            opt=opt,
            scheduler=scheduler,
            logger=logger,
            max_num_batches=max_num_batches
        )

        if use_val_set and get_rank() == 0:
            with torch.no_grad(), accelerator.autocast():
                translator.eval()
                val_res = {}
                _, trans = eval_loop_(cfg, translator, encoders, valloader, device=accelerator.device)
                for target_flag, d in trans.items():
                    for flag, res in d.items():
                        for k, v in res.items():
                            val_res[f"val_{flag}_{target_flag}_{k}"] = v
                
                wandb.log(val_res)

                if cfg.targeted:
                    print(val_res[f"val_{cfg.src_emb}_{cfg.tgt_emb}_{'cos'}"])
                    if early_stopper.early_stop(val_res[f"val_{cfg.src_emb}_{cfg.tgt_emb}_{'cos'}"]):
                        print("Stopping early! Saving previous model...")
                        break
                    elif early_stopper.counter == 0:
                        best_model = accelerator.unwrap_model(translator).state_dict().copy()

    with open(cfg.save_dir + 'config.toml', 'w') as f:
        toml.dump(cfg.__dict__, f)
    # save model
    best_model = best_model if best_model is not None else accelerator.unwrap_model(translator).state_dict()
    torch.save(best_model, model_save_dir)

    # eval
    cfg.use_good_queries = 0
    cfg.dataset = "NanoNQ"
    cfg.max_seq_length = cfg.max_seq_length
    cfg.batch_size = cfg.bs
    cfg.train_path = cfg.save_dir
    metrics = eval_model(cfg=cfg, translator=translator)
    wandb.log({ f"eval/k": v for k,v in metrics.items() })

if __name__ == "__main__":
    main()
