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
import torch.nn.functional as F

from translators.Discriminator import Discriminator

# from eval import eval_model
from utils.collate import MultiEncoderCollator
from utils.dist import get_rank, get_world_size
from utils.eval_utils import EarlyStopper, eval_loop_
from utils.gan import VanillaGAN
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings, process_batch
from utils.train_utils import rec_loss_fn, trans_loss_fn, vsp_loss_fn, get_grad_norm
from utils.wandb_logger import Logger

import matplotlib.pyplot as plt
import seaborn as sns


def training_loop_(
    save_dir, accelerator, gan, translator, disc, sup_dataloader, unsup_dataloader, sup_encs, unsup_enc, cfg, opt, scheduler, disc_opt, logger=None, max_num_batches=None
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
            assert len(set(sup_batch.keys()).intersection(unsup_batch.keys())) == 0

            ins = {**process_batch(cfg, sup_batch, sup_encs, device), **process_batch(cfg, unsup_batch, unsup_enc, device)}
            if cfg.add_noise:
                min_noise_pow = -10
                max_noise_pow = -1
            else:
                min_noise_pow = 0
                max_noise_pow = 0

            recons, translations = (
                accelerator.unwrap_model(translator).forward(ins, max_noise_pow, min_noise_pow)
            )
            real_data = ins[cfg.sup_emb] # formerly sup_to_sup
            fake_data = translations[cfg.sup_emb][cfg.unsup_emb].detach() # formerly unsup_to_sup
            
            disc_loss, disc_acc_real, disc_acc_fake = gan.step_discriminator(
                real_data=real_data, 
                fake_data=fake_data
            )
            gen_loss, gen_acc = gan.step_generator(
                fake_data=fake_data
            )

            rec_loss = rec_loss_fn(ins, recons, logger)
            if cfg.loss_coefficient_cc > 0:
                cc_keys = list(translations.keys())
                random.shuffle(cc_keys)
                cc_translations = dict(
                    itertools.chain(*[{ k1: v.detach()  for v in translations[k1].values()}.items() for k1 in cc_keys]))
                cc_recons, cc_translations = (
                    accelerator.unwrap_model(translator).forward(cc_translations, max_noise_pow, min_noise_pow)
                )
                cc_rec_loss = rec_loss_fn(ins, cc_recons, logger, prefix="cc_")
                cc_trans_loss = trans_loss_fn(ins, cc_translations, logger, prefix="cc_")
            else:
                cc_rec_loss = torch.tensor(0.0)
                cc_trans_loss = torch.tensor(0.0)

            if cfg.loss_coefficient_vsp > 0:
                vsp_loss = vsp_loss_fn(ins, cc_translations, logger)
            else:
                vsp_loss = torch.tensor(0.0)

            loss = (
                  (rec_loss * cfg.loss_coefficient_rec)
                + (vsp_loss * cfg.loss_coefficient_vsp)
                + ((cc_rec_loss + cc_trans_loss) * cfg.loss_coefficient_cc)
                + (gen_loss * cfg.loss_coefficient_adv)
            )
            opt.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(translator.parameters(), cfg.max_grad_norm)
            grad_norm = get_grad_norm(translator)

            opt.step()
            if not hasattr(cfg, 'no_scheduler') or not cfg.no_scheduler:
                scheduler.step()

            metrics = {
                "disc_loss": disc_loss.item(),
                "rec_loss": rec_loss.item(),
                "vsp_loss": vsp_loss.item(),
                "cc_rec_loss": cc_rec_loss.item(),
                "cc_trans_loss": cc_trans_loss.item(),
                "gen_loss": gen_loss.item(),
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "learning_rate": opt.param_groups[0]["lr"],
                "disc_acc_real": disc_acc_real,
                "disc_acc_fake": disc_acc_fake,
                "gen_acc": gen_acc,
            }

            for metric, value in metrics.items():
                logger.logkv(metric, value)
            logger.dumpkvs(force=(hasattr(cfg, 'force_dump') and cfg.force_dump))
            if get_rank() == 0:
                dataloader_pbar.set_postfix(metrics)

        if (i + 1) % cfg.save_every == 0:
            with open(save_dir + 'config.toml', 'w') as f:
                toml.dump(cfg.__dict__, f)
            torch.save(accelerator.unwrap_model(translator).state_dict(), model_save_dir)



# TODO: change embs to supervised_emb
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'configs/{argv[1]}.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**cfg['general'], **cfg['train'], **cfg['discriminator'], **cfg['logging'], **unknown_cfg})

    use_val_set = hasattr(cfg, 'val_size')

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision,
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

    sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision='bf16' == cfg.mixed_precision)}
    encoder_dims = {cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])}
    translator = load_n_translator(cfg, encoder_dims)

    model_save_dir = os.path.join(save_dir, 'model.pt')
    disc_save_dir = os.path.join(save_dir, 'disc.pt')

    os.makedirs(save_dir, exist_ok=True)

    # if hasattr(cfg, 'freeze_params') and cfg.freeze_params:
    #     for param in translator.parameters():
    #         param.requires_grad = False

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision='bf16' == cfg.mixed_precision)}
    unsup_dim = {cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb])}
    translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])

    assert cfg.unsup_emb not in sup_encs
    assert cfg.unsup_emb in translator.in_adapters
    assert cfg.unsup_emb in translator.out_adapters

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

    disc_norm = 'spectral' if hasattr(cfg, 'use_spectral') and cfg.use_spectral else None
    disc = Discriminator(768, cfg.disc_dim, cfg.disc_depth, cfg.use_residual, norm_style=disc_norm)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps)
    if cfg.finetune_mode:
        assert hasattr(cfg, 'load_dir')
        print(f"Loading models from {cfg.load_dir}...")
        translator.load_state_dict(torch.load(cfg.load_dir + 'model.pt', map_location='cpu'), strict=False)
        disc.load_state_dict(torch.load(cfg.load_dir + 'disc.pt', map_location='cpu'))

    translator, opt, scheduler, sup_dataloader, unsup_dataloader, disc, disc_opt = accelerator.prepare(
        translator, opt, scheduler, sup_dataloader, unsup_dataloader, disc, disc_opt
    )

    best_model = None
    best_disc = None
    early_stopper = EarlyStopper(
        patience=cfg.patience if hasattr(cfg, 'patience') else 1,
        min_delta=cfg.delta if hasattr(cfg, 'delta') else 0
    )

    for epoch in range(max_num_epochs):
        if use_val_set and get_rank() == 0:
            with torch.no_grad(), accelerator.autocast():
                translator.eval()
                val_res = {}
                recons, trans, heatmap = eval_loop_(cfg, translator, {**sup_encs, **unsup_enc}, valloader, device=accelerator.device)
                for flag, res in recons.items():
                    for k, v in res.items():
                        if k == 'cos':
                            val_res[f"val/rec_{flag}_{k}"] = v
                for target_flag, d in trans.items():
                    for flag, res in d.items():
                        for k, v in res.items():
                            if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                                continue
                            val_res[f"val/{flag}_{target_flag}_{k}"] = v
                
                if heatmap is not None:
                    for k, v in heatmap.items():
                        if k == 'heatmap':
                            v = wandb.Image(v)
                        val_res[f"val/{k}"] = v
                wandb.log(val_res)

                # if early_stopper.early_stop(val_res['stop_cond']):
                #     print("Stopping early! Saving previous model...")
                #     break
                # else:
                #     best_model = translator.state_dict().copy()
                #     best_disc = disc.state_dict().copy()
        
        gan = VanillaGAN(
            cfg=cfg, 
            generator=translator, 
            discriminator=disc, 
            generator_opt=opt, 
            discriminator_opt=disc_opt, 
            accelerator=accelerator
        )
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
            gan=gan,
            disc=disc,
            sup_dataloader=sup_dataloader,
            unsup_dataloader=unsup_dataloader,
            sup_encs=sup_encs,
            unsup_enc=unsup_enc,
            cfg=cfg,
            opt=opt,
            scheduler=scheduler,
            disc_opt=disc_opt,
            logger=logger,
            max_num_batches=max_num_batches
        )

    with open(save_dir + 'config.toml', 'w') as f:
        toml.dump(cfg.__dict__, f)
    # save model
    best_model = best_model if best_model is not None else accelerator.unwrap_model(translator).state_dict()
    best_disc = best_disc if best_disc is not None else disc.state_dict()
    torch.save(best_model, model_save_dir)
    torch.save(best_disc, disc_save_dir)

    # # eval
    # cfg.use_good_queries = 1
    # cfg.dataset = "NanoNQ"
    # cfg.max_seq_length = cfg.max_seq_length
    # cfg.batch_size = cfg.bs
    # cfg.train_path = save_dir
    # metrics = eval_model(cfg=cfg, translator=translator)
    # wandb.log({ f"eval/k": v for k,v in metrics.items() })

if __name__ == "__main__":
    main()
