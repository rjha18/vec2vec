import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate
from tqdm import tqdm
import wandb

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from translators.Discriminator import Discriminator

# from eval import eval_model
from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.dist import get_rank, get_world_size
from utils.eval_utils import eval_loop_
from utils.gan import LeastSquaresGAN, RelativisticGAN, VanillaGAN
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings, process_batch
from utils.train_utils import rec_loss_fn, trans_loss_fn, vsp_loss_fn, get_grad_norm
from utils.wandb_logger import Logger


def training_loop_(
    save_dir, accelerator, gan, sup_gan, latent_gan, similarity_gan, translator, sup_dataloader, unsup_dataloader, sup_encs, unsup_enc, cfg, opt, scheduler, logger=None, max_num_batches=None
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

    translator.train()
    for i, (sup_batch, unsup_batch) in enumerate(dataloader_pbar):
        if max_num_batches is not None and i >= max_num_batches:
            print(f"Early stopping at {i} batches")
            break
        with accelerator.accumulate(translator), accelerator.autocast():
            assert len(set(sup_batch.keys()).intersection(unsup_batch.keys())) == 0
            ins = {
                **process_batch(cfg, sup_batch, sup_encs, device), 
                **process_batch(cfg, unsup_batch, unsup_enc, device)
            }
            recons, translations, reps = translator(
                ins, noise_level=cfg.noise_level, include_reps=True
            )

            # discriminator
            disc_r1_penalty, disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc = gan.step(
                real_data=ins[cfg.unsup_emb] + torch.randn_like(ins[cfg.unsup_emb], device=ins[cfg.unsup_emb].device) * cfg.noise_level,
                fake_data=translations[cfg.unsup_emb][cfg.sup_emb] + torch.randn_like(translations[cfg.unsup_emb][cfg.sup_emb], device=translations[cfg.unsup_emb][cfg.sup_emb].device) * cfg.noise_level
            )

            sup_disc_r1_penalty, sup_disc_loss, sup_gen_loss, sup_disc_acc_real, sup_disc_acc_fake, sup_gen_acc = sup_gan.step(
                real_data=ins[cfg.sup_emb] + torch.randn_like(ins[cfg.sup_emb], device=ins[cfg.sup_emb].device) * cfg.noise_level,
                fake_data=translations[cfg.sup_emb][cfg.unsup_emb] + torch.randn_like(translations[cfg.sup_emb][cfg.unsup_emb], device=translations[cfg.sup_emb][cfg.unsup_emb].device) * cfg.noise_level,
            )

            # latent discriminator
            latent_disc_r1_penalty, latent_disc_loss, latent_gen_loss, latent_disc_acc_real, latent_disc_acc_fake, latent_gen_acc = latent_gan.step(
                real_data=reps[cfg.sup_emb],
                fake_data=reps[cfg.unsup_emb]
            )

            # similarity discriminator
            if cfg.loss_coefficient_similarity_gen > 0:
                real_sims_A = ins[cfg.sup_emb] @ ins[cfg.sup_emb].T
                fake_sims_A = (
                    translations[cfg.sup_emb][cfg.unsup_emb] @ translations[cfg.sup_emb][cfg.unsup_emb].T
                )
                real_sims_B = ins[cfg.unsup_emb] @ ins[cfg.unsup_emb].T
                fake_sims_B = (
                    translations[cfg.unsup_emb][cfg.sup_emb] @ translations[cfg.unsup_emb][cfg.sup_emb].T
                )
                similarity_r1_penalty, similarity_disc_loss, similarity_gen_loss, similarity_disc_acc_real, similarity_disc_acc_fake, similarity_gen_acc = similarity_gan.step(
                    real_data=torch.cat([real_sims_A, real_sims_B], dim=1),
                    fake_data=torch.cat([fake_sims_A, fake_sims_B], dim=1)
                )
            else:
                similarity_r1_penalty = torch.tensor(0.0)
                similarity_disc_loss = torch.tensor(0.0)
                similarity_gen_loss = torch.tensor(0.0)
                similarity_disc_acc_real = 0.0
                similarity_disc_acc_fake = 0.0
                similarity_gen_acc = 0.0
            
            rec_loss = rec_loss_fn(ins, recons, logger)
            ins_reversed = {
                cfg.sup_emb: ins[cfg.unsup_emb],
                cfg.unsup_emb: ins[cfg.sup_emb],
            }
            translations_as_recons = {
                cfg.sup_emb: translations[cfg.unsup_emb][cfg.sup_emb],
                cfg.unsup_emb: translations[cfg.sup_emb][cfg.unsup_emb],
            }
            reverse_rec_loss = rec_loss_fn(ins_reversed, translations_as_recons, logger, prefix="reverse_")

            recons_as_translations = { 
                in_name: { in_name: val } for in_name, val in recons.items() 
            }
            vsp_loss = vsp_loss_fn(ins, recons_as_translations, logger)
            if (cfg.loss_coefficient_cc_rec > 0) or (cfg.loss_coefficient_cc_trans > 0):
                cc_ins = {}
                for out_flag in translations.keys():
                    in_flag = random.choice(list(translations[out_flag].keys()))
                    cc_ins[out_flag] = translations[out_flag][in_flag].detach()
                cc_recons, cc_translations = translator(cc_ins)
                cc_rec_loss = rec_loss_fn(ins, cc_recons, logger, prefix="cc_")
                cc_trans_loss = trans_loss_fn(ins, cc_translations, logger, prefix="cc_")
                cc_vsp_loss = vsp_loss_fn(ins, cc_translations, logger)
            else:
                cc_rec_loss = torch.tensor(0.0)
                cc_trans_loss = torch.tensor(0.0)
                cc_vsp_loss = torch.tensor(0.0)

            loss = (
                + (rec_loss * cfg.loss_coefficient_rec)
                + (reverse_rec_loss * cfg.loss_coefficient_reverse_rec)
                + (vsp_loss * cfg.loss_coefficient_vsp)
                + (cc_vsp_loss * cfg.loss_coefficient_cc_vsp)
                + (cc_rec_loss * cfg.loss_coefficient_cc_rec)
                + (cc_trans_loss * cfg.loss_coefficient_cc_trans)
                + (gen_loss * cfg.loss_coefficient_gen)
                + (sup_gen_loss * cfg.loss_coefficient_gen)
                + (latent_gen_loss * cfg.loss_coefficient_latent_gen)
                + (similarity_gen_loss * cfg.loss_coefficient_similarity_gen)
            )
            exit_on_nan(loss)
            opt.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(translator.parameters(), cfg.max_grad_norm)
            grad_norm_generator = get_grad_norm(translator)
            grad_norm_discriminator = get_grad_norm(gan.discriminator)
            grad_norm_sup_discriminator = get_grad_norm(sup_gan.discriminator)
            grad_norm_latent_discriminator = get_grad_norm(latent_gan.discriminator)
            grad_norm_similarity_discriminator = get_grad_norm(similarity_gan.discriminator)

            opt.step()
            scheduler.step()

            metrics = {
                "disc_loss": disc_loss.item(),
                "disc_r1_penalty": disc_r1_penalty.item(),
                "sup_disc_loss": sup_disc_loss.item(),
                "sup_disc_r1_penalty": sup_disc_r1_penalty.item(),
                "latent_disc_loss": latent_disc_loss.item(),
                "latent_disc_r1_penalty": latent_disc_r1_penalty.item(),
                "similarity_disc_loss": similarity_disc_loss.item(),
                "similarity_r1_penalty": similarity_r1_penalty.item(),
                "rec_loss": rec_loss.item(),
                "reverse_rec_loss": reverse_rec_loss.item(),
                "vsp_loss": vsp_loss.item(),
                "cc_vsp_loss": cc_vsp_loss.item(),
                "cc_rec_loss": cc_rec_loss.item(),
                "cc_trans_loss": cc_trans_loss.item(),
                "gen_loss": gen_loss.item(),
                "sup_gen_loss": sup_gen_loss.item(),
                "latent_gen_loss": latent_gen_loss.item(),
                "similarity_gen_loss": similarity_gen_loss.item(),
                "loss": loss.item(),
                "grad_norm_generator": grad_norm_generator,
                "grad_norm_discriminator": grad_norm_discriminator,
                "grad_norm_sup_discriminator": grad_norm_sup_discriminator,
                "grad_norm_latent_discriminator": grad_norm_latent_discriminator,
                "grad_norm_similarity_discriminator": grad_norm_similarity_discriminator,
                "learning_rate": opt.param_groups[0]["lr"],
                "disc_acc_real": disc_acc_real,
                "disc_acc_fake": disc_acc_fake,
                "latent_disc_acc_real": latent_disc_acc_real,
                "latent_disc_acc_fake": latent_disc_acc_fake,
                "gen_acc": gen_acc,
                "sup_disc_acc_real": sup_disc_acc_real,
                "sup_disc_acc_fake": sup_disc_acc_fake,
                "sup_gen_acc": sup_gen_acc,
                "similarity_disc_acc_real": similarity_disc_acc_real,
                "similarity_disc_acc_fake": similarity_disc_acc_fake,
                "similarity_gen_acc": similarity_gen_acc,
            }

            for metric, value in metrics.items():
                logger.logkv(metric, value)
            logger.dumpkvs(force=(hasattr(cfg, 'force_dump') and cfg.force_dump))
            if get_rank() == 0:
                dataloader_pbar.set_postfix(metrics)

    with open(save_dir + 'config.toml', 'w') as f:
        toml.dump(cfg.__dict__, f)
    torch.save(accelerator.unwrap_model(translator).state_dict(), model_save_dir)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'configs/{argv[1]}.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**{k: v for d in cfg.values() for k, v in d.items()}, **unknown_cfg})

    if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        cfg.mixed_precision = 'fp16'
        cfg.gradient_accumulation_steps = 1
        print("Note: bf16 is not available on this hardware! Reverting to fp16 and setting accumulation steps to 1.")

    # set seeds
    random.seed(cfg.seed + get_rank())
    torch.manual_seed(cfg.seed + get_rank())
    np.random.seed(cfg.seed + get_rank())
    torch.cuda.manual_seed(cfg.seed + get_rank())

    use_val_set = hasattr(cfg, 'val_size')

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )
    # https://github.com/huggingface/transformers/issues/26548
    accelerator.dataloader_config.dispatch_batches = False

    if hasattr(cfg, 'force_wandb_name') and cfg.force_wandb_name:
        save_dir = cfg.save_dir.format(cfg.wandb_name)
    else:
        cfg.wandb_name = ','.join([f"{k[0]}:{v}" for k, v in unknown_cfg.items()]) if unknown_cfg else cfg.wandb_name
        save_dir = cfg.save_dir.format(cfg.latent_dims if hasattr(cfg, 'latent_dims') else cfg.wandb_name)

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=(cfg.wandb_project is None) or not (cfg.use_wandb),
        config=cfg,
    )

    print("Running Experiment:", cfg.wandb_name)

    dset = load_streaming_embeddings(cfg.dataset)

    sup_encs = {
        cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)
    }
    encoder_dims = {
        cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])
    }
    translator = load_n_translator(cfg, encoder_dims)

    model_save_dir = os.path.join(save_dir, 'model.pt')
    disc_save_dir = os.path.join(save_dir, 'disc.pt')

    os.makedirs(save_dir, exist_ok=True)

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
    print("Number of *trainable* parameters:", sum(p.numel() for p in translator.parameters() if p.requires_grad))
    print("Number of training datapoints:", len(dset))
    print(translator)

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=(cfg.wandb_project is None) or not (cfg.use_wandb),
        config=cfg,
    )

    num_workers = get_num_proc()
    print(f"Rank {get_rank()} using {num_workers} workers and {len(dset)} datapoints")
    if get_world_size() > 1:
        max_num_datapoints = len(dset) - (len(dset) % (cfg.bs * get_world_size()))
        dset = dset.select(range(max_num_datapoints))
        print(f"[Filtered] Rank {get_rank()} now using {len(dset)} datapoints")

    dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    dset = dset_dict["train"]
    valset = dset_dict["test"]
    if hasattr(cfg, 'num_points'):
        supset = dset.shuffle(seed=cfg.train_dataset_seed + 1).select(range(min(cfg.num_points, len(dset))))
        unsupset = dset.shuffle(seed=cfg.train_dataset_seed + 2).select(range(min(cfg.num_points, len(dset))))

    supset = MultiencoderTokenizedDataset(
        dataset=supset,
        encoders=sup_encs,
        n_embs_per_batch=cfg.n_embs_per_batch,
        batch_size=cfg.bs,
        max_length=cfg.max_seq_length,
        seed=cfg.seed,
    )
    unsupset = MultiencoderTokenizedDataset(
        dataset=unsupset,
        encoders=unsup_enc,
        n_embs_per_batch=1,
        batch_size=cfg.bs,
        max_length=cfg.max_seq_length,
        seed=cfg.seed,
    )

    sup_dataloader = DataLoader(
        supset,
        batch_size=cfg.bs,
        num_workers=num_workers // 2,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=None,
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    unsup_dataloader = DataLoader(
        unsupset,
        batch_size=cfg.bs,
        num_workers=num_workers // 2,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=None,
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )

    if use_val_set:
        valset = MultiencoderTokenizedDataset(
            dataset=valset,
            encoders={ **unsup_enc, **sup_encs },
            n_embs_per_batch=2,
            batch_size=cfg.val_bs,
            max_length=cfg.max_seq_length,
            seed=cfg.seed,
        )
        valloader = DataLoader(
            valset,
            batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=(8 if num_workers > 0 else None),
            collate_fn=TokenizedCollator(),
            drop_last=True,
        )
        valloader = accelerator.prepare(valloader)

    opt = torch.optim.Adam(translator.parameters(), lr=cfg.lr, fused=False, betas=(0.5, 0.999))
    
    ######################################################################################
    disc = Discriminator(
        latent_dim=translator.in_adapters[cfg.unsup_emb].in_dim,
        discriminator_dim=cfg.disc_dim,
        depth=cfg.disc_depth,
        weight_init=cfg.weight_init
    )
    disc_opt = torch.optim.Adam(disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps, betas=(0.5, 0.999))

    cfg.num_disc_params = sum(x.numel() for x in disc.parameters())
    print(f"Number of discriminator parameters:", cfg.num_disc_params)
    ######################################################################################
    sup_disc = Discriminator(
        latent_dim=translator.in_adapters[cfg.sup_emb].in_dim,
        discriminator_dim=cfg.disc_dim, 
        depth=cfg.disc_depth, 
    )
    sup_disc_opt = torch.optim.Adam(sup_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps, betas=(0.5, 0.999))

    cfg.num_sup_disc_params = sum(x.numel() for x in sup_disc.parameters())
    print(f"Number of supervised discriminator parameters:", cfg.num_sup_disc_params)
    print(sup_disc)
    ######################################################################################
    latent_disc = Discriminator(
        latent_dim=cfg.d_adapter,
        discriminator_dim=cfg.disc_dim,
        depth=cfg.disc_depth,
        weight_init=cfg.weight_init
    )
    latent_disc_opt = torch.optim.RMSprop(latent_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps)
    cfg.num_latent_disc_params = sum(x.numel() for x in latent_disc.parameters())
    print(f"Number of latent discriminator parameters:", cfg.num_latent_disc_params)
    print(latent_disc)
    latent_disc_opt = torch.optim.Adam(latent_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps, betas=(0.5, 0.999))
    ######################################################################################
    similarity_disc = Discriminator(
        latent_dim=(cfg.bs * 2),
        discriminator_dim=cfg.disc_dim,
        depth=cfg.disc_depth,
        weight_init=cfg.weight_init
    )
    similarity_disc_opt = torch.optim.RMSprop(similarity_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps)
    cfg.num_similarity_disc_params = sum(x.numel() for x in similarity_disc.parameters())
    print(f"Number of similarity discriminator parameters:", cfg.num_similarity_disc_params)
    print(similarity_disc)
    similarity_disc_opt = torch.optim.Adam(similarity_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps, betas=(0.5, 0.999))
    ######################################################################################

    max_num_epochs = int(np.ceil(cfg.epochs))
    steps_per_epoch = len(supset) // cfg.bs
    total_steps = steps_per_epoch * cfg.epochs / cfg.gradient_accumulation_steps
    warmup_length = (cfg.warmup_length if hasattr(cfg, 'warmup_length') else 100) * get_world_size()

    def lr_lambda(step):
        if step < warmup_length:
            return min(1, step / warmup_length)
        else:
            if hasattr(cfg, 'no_scheduler') and cfg.no_scheduler:
                return 1
            return 1 - (step - warmup_length) / max(1, total_steps - warmup_length)

    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)
    disc_scheduler = LambdaLR(disc_opt, lr_lambda=lr_lambda)
    sup_disc_scheduler = LambdaLR(sup_disc_opt, lr_lambda=lr_lambda)
    latent_disc_scheduler = LambdaLR(latent_disc_opt, lr_lambda=lr_lambda)
    similarity_disc_scheduler = LambdaLR(similarity_disc_opt, lr_lambda=lr_lambda)

    if cfg.finetune_mode:
        assert hasattr(cfg, 'load_dir')
        print(f"Loading models from {cfg.load_dir}...")
        translator.load_state_dict(torch.load(cfg.load_dir + 'model.pt', map_location='cpu'), strict=False)
        disc.load_state_dict(torch.load(cfg.load_dir + 'disc.pt', map_location='cpu'))

    translator, opt, scheduler = accelerator.prepare(translator, opt, scheduler)
    disc, disc_opt, disc_scheduler = accelerator.prepare(disc, disc_opt, disc_scheduler)
    sup_disc, sup_disc_opt, sup_disc_scheduler = accelerator.prepare(sup_disc, sup_disc_opt, sup_disc_scheduler)
    latent_disc, latent_disc_opt, latent_disc_scheduler = accelerator.prepare(latent_disc, latent_disc_opt, latent_disc_scheduler)
    similarity_disc, similarity_disc_opt, similarity_disc_scheduler = accelerator.prepare(
        similarity_disc, similarity_disc_opt, similarity_disc_scheduler
    )
    sup_dataloader, unsup_dataloader = accelerator.prepare(sup_dataloader, unsup_dataloader)

    best_model = None
    best_disc = None

    if cfg.gan_style == "vanilla":
        gan_cls = VanillaGAN
    elif cfg.gan_style == "least_squares":
        gan_cls = LeastSquaresGAN
    elif cfg.gan_style == "relativistic":
        gan_cls = RelativisticGAN
    else:
        raise ValueError(f"Unknown GAN style: {cfg.gan_style}")
    latent_gan = gan_cls(
        cfg=cfg,
        generator=translator,
        discriminator=latent_disc,
        discriminator_opt=latent_disc_opt,
        discriminator_scheduler=latent_disc_scheduler,
        accelerator=accelerator,
    )
    similarity_gan = gan_cls(
        cfg=cfg,
        generator=translator,
        discriminator=similarity_disc,
        discriminator_opt=similarity_disc_opt,
        discriminator_scheduler=similarity_disc_scheduler,
        accelerator=accelerator,
    )
    gan = gan_cls(
        cfg=cfg,
        generator=translator,
        discriminator=disc,
        discriminator_opt=disc_opt,
        discriminator_scheduler=disc_scheduler,
        accelerator=accelerator,
    )
    sup_gan = gan_cls(
        cfg=cfg,
        generator=translator,
        discriminator=sup_disc,
        discriminator_opt=sup_disc_opt,
        discriminator_scheduler=sup_disc_scheduler,
        accelerator=accelerator
    )
    for epoch in range(max_num_epochs):
        if use_val_set and get_rank() == 0:
            with torch.no_grad(), accelerator.autocast():
                translator.eval()
                val_res = {}
                recons, trans, heatmap_dict = eval_loop_(cfg, translator, {**sup_encs, **unsup_enc}, valloader, device=accelerator.device)
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

                if len(heatmap_dict) > 0:
                    for k,v in heatmap_dict.items():
                        if k in ["heatmap", "heatmap_softmax"]:
                            v = wandb.Image(v)
                            val_res[f"val/{k}"] = v
                        else:
                            val_res[f"val/{k} (avg. {cfg.top_k_batches} batches)"] = v
                wandb.log(val_res)
                translator.train()

        max_num_batches = None
        print(f"Epoch", epoch, "max_num_batches", max_num_batches, "max_num_epochs", max_num_epochs)
        if epoch + 1 >= max_num_epochs:
            max_num_batches = max(1, (cfg.epochs - epoch) * len(supset) // cfg.bs // get_world_size())
            print(f"Setting max_num_batches to {max_num_batches}")

        training_loop_(
            save_dir=save_dir,
            accelerator=accelerator,
            translator=translator,
            gan=gan,
            sup_gan=sup_gan,
            latent_gan=latent_gan,
            similarity_gan=similarity_gan,
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
    best_disc = best_disc if best_disc is not None else disc.state_dict()
    torch.save(best_model, model_save_dir)
    torch.save(best_disc, disc_save_dir)


if __name__ == "__main__":
    main()
