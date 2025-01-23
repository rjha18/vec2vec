import os
from sys import argv
from tqdm import tqdm
from types import SimpleNamespace
from eval_utils import eval_loop_, text_loop_
from translators.Discriminator import Discriminator
from utils import *
import wandb
import toml

import numpy as np
import torch
import torch.nn.functional as F


from models.kernels import mse
from models.logging import Logger
from embeddings import load_and_process_embeddings


def loss_fn(X, Y, X_recons, Y_recons, logger, mse_scales):
    X_mse_scale, Y_mse_scale = mse_scales
    loss = logger.logkv("X_train_recons", X_mse_scale * mse(X_recons, X))
    loss += logger.logkv("Y_train_recons", Y_mse_scale * mse(Y_recons, Y))
    return loss

def training_loop_(
    ae, disc, X_iter, Y_iter, cfg, mse_scales, autocast_ctx_manager, logger=None, pbar=None,
):
    if logger is None:
        logger = Logger(dummy=True)

    scaler = torch.cuda.amp.GradScaler()

    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr, eps=cfg.eps, fused=True)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps)

    for _, (x, y) in enumerate(zip(X_iter, Y_iter)):
        res = {}
        x, y = x.cuda(), y.cuda()
        x_n = x + cfg.noise * torch.randn_like(x)
        y_n = y + cfg.noise * torch.randn_like(y)

        if cfg.disc_coef > 0:
            # Discriminator training
            with autocast_ctx_manager:
                recons_A, recons_B, rep_A, rep_B = ae(x_n, y_n, rep=cfg.rep)
                d_A, d_B = disc(rep_A), disc(rep_B)

                disc_ce_A = F.binary_cross_entropy_with_logits(d_A, torch.zeros((x.shape[0], 1), device=x.device))
                disc_ce_B = F.binary_cross_entropy_with_logits(d_B, torch.ones((y.shape[0], 1), device=y.device)) * cfg.smooth
                disc_loss = disc_ce_A + disc_ce_B
                disc_loss *= cfg.disc_coef

            disc_opt.zero_grad()
            scaler.scale(disc_loss).backward()
            res['d_loss'] = disc_loss.item()
            res['x_mse'] = mse(recons_A, x).item()
            res['y_mse'] = mse(recons_B, y).item()
            scaler.unscale_(disc_opt)
            scaler.step(disc_opt)
            scaler.update()

        # Autoencoder forward training
        with autocast_ctx_manager:
            # compute adv_loss
            recons_A, recons_B, rep_A, rep_B = ae(x_n, y_n, rep=cfg.rep)
            d_A, d_B = disc(rep_A), disc(rep_B)

            adv_loss_A = F.binary_cross_entropy_with_logits(d_A, torch.ones((x.shape[0], 1), device=x.device)) * cfg.smooth
            adv_loss_B = F.binary_cross_entropy_with_logits(d_B, torch.zeros((y.shape[0], 1), device=y.device))
            adv_loss = adv_loss_A + adv_loss_B

            loss = loss_fn(x, y, recons_A, recons_B, logger, mse_scales)
            loss += cfg.adv_coef * adv_loss


        opt.zero_grad()
        scaler.scale(loss).backward()
        res['loss'] = loss.item()
        res['adv_loss'] = adv_loss.item()
        ae.unit_norm_decoder_()
        ae.unit_norm_decoder_grad_adjustment_()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()

        # Autoencoder backward training
        with autocast_ctx_manager:
            x_prime, y_prime, rep_phi_B, rep_phi_A = ae.cross_flow(x_n, y_n, rep=cfg.rep)
            d_A, d_B = disc(rep_phi_A), disc(rep_phi_B)

            back_adv_loss_A = F.binary_cross_entropy_with_logits(d_A, torch.ones((x.shape[0], 1), device=x.device)) * cfg.smooth
            back_adv_loss_B = F.binary_cross_entropy_with_logits(d_B, torch.zeros((y.shape[0], 1), device=y.device))
            back_adv_loss = back_adv_loss_A + back_adv_loss_B

            back_loss = mse_scales[0] * mse(x_prime, x) + mse_scales[1] * mse(y_prime, y)
            back_loss += cfg.adv_coef * back_adv_loss

        opt.zero_grad()
        scaler.scale(back_loss).backward()
        res['back_loss'] = back_loss.item()
        ae.unit_norm_decoder_()
        ae.unit_norm_decoder_grad_adjustment_()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()

        wandb.log(res)        
        logger.dumpkvs()
        if pbar is not None:
            pbar.set_postfix(res)
            pbar.update(1)


def main():
    cfg = toml.load(f'configs/{argv[1]}.toml')
    cfg = SimpleNamespace(**cfg['general'], **cfg['train'], **cfg['discriminator'], **cfg['logging'])

    if len(argv) >= 3:
        print("Overriding config with command line arguments")
        cfg.latent_dims = int(argv[2])

    save_dir = cfg.save_dir.format(cfg.latent_dims)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    cfg.wandb_name = f"{cfg.latent_dims}_{cfg.wandb_name}_{cfg.d_adapter}"
    print("Running Experiment:", cfg.wandb_name)

    X, X_mask = load_and_process_embeddings(
        cfg.dset, cfg.emb1, cfg.num_points, cfg.seed, cfg.normalize, 'train', 32, False, 'cuda', 
    )

    Y, Y_mask = load_and_process_embeddings(
        cfg.dset, cfg.emb2, cfg.num_points, cfg.seed+1, cfg.normalize, 'train', 32, False, 'cuda'
    )

    np.save(save_dir + 'X_mask.npy', X_mask)
    np.save(save_dir + 'Y_mask.npy', Y_mask)

    X_iter, Y_iter = get_loaders([X, Y], cfg.bs, True)

    X_stats_sample = X[:cfg.stats_sample_size]['text_embeddings']
    Y_stats_sample = Y[:cfg.stats_sample_size]['text_embeddings']

    X_train, Y_train = get_val_sets([X_mask, Y_mask], cfg, cfg.seed + 2, test_flag=False, keep_in_memory=False)
    X_train_dl, Y_train_dl = get_loaders([X_train, Y_train], cfg.val_bs, False)

    X_test, Y_test = get_val_sets([X_mask, Y_mask], cfg, cfg.seed + 3, test_flag=True, keep_in_memory=False)
    X_test_dl, Y_test_dl = get_loaders([X_test, Y_test], cfg.val_bs, False)

    if cfg.metrics_every > 0:
        X_sub_train, Y_sub_train = get_text_sets([X_train, Y_train], cfg.metrics_size, cfg.seed + 4)
        X_sub_test, Y_sub_test = get_text_sets([X_test, Y_test], cfg.metrics_size, cfg.seed + 5)
        idx_dataloader = get_text_loader(cfg.metrics_size)
        corrector = get_corrector()

    ae = load_dense(cfg)

    if hasattr(cfg, 'load_from_epoch'):
        ae.load_state_dict(torch.load(save_dir + f'model_{cfg.load_from_epoch}.pt'))
    ae.cuda()

    disc_dims = cfg.latent_dims if cfg.rep == 'latent' else cfg.d_adapter
    disc = Discriminator(disc_dims, cfg.disc_dim, cfg.disc_depth)
    disc.cuda()

    print("Number of parameters:", sum(x.numel() for x in ae.parameters()))

    X_mse_scale = (
        1 / ((X_stats_sample.float().mean(dim=0) - X_stats_sample.float()) ** 2).mean()
    ).item()

    Y_mse_scale = (
        1 / ((Y_stats_sample.float().mean(dim=0) - Y_stats_sample.float()) ** 2).mean()
    ).item()

    mse_scales = (X_mse_scale, Y_mse_scale)
    print("MSE scales:", *mse_scales)

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=cfg.wandb_project is None,
        config=cfg,
    )
    autocast_ctx_manager = torch.cuda.amp.autocast()

    pbar = tqdm(range(cfg.epochs * len(X_iter)))
    for i in range(cfg.epochs):
        training_loop_(ae, disc, X_iter, Y_iter, cfg, mse_scales, autocast_ctx_manager, logger=logger, pbar=pbar)
        
        val_metrics = {}
        with torch.no_grad():
            phi_res = eval_loop_(ae, X_train_dl, Y_train_dl, autocast_ctx_manager, rep=cfg.rep)
            val_metrics.update({f"train_{k}": v for k, v in phi_res.items()})

            phi_res = eval_loop_(ae, X_test_dl, Y_test_dl, autocast_ctx_manager, rep=cfg.rep)
            val_metrics.update({f"test_{k}": v for k, v in phi_res.items()})

            if (i + 1) % cfg.metrics_every == 0:
                bleus, f1s = text_loop_(ae, corrector, X_sub_train, Y_sub_train, idx_dataloader, rep=cfg.rep)
                
                val_metrics.update({f"train_b{k}": v for k, v in bleus.items()})
                val_metrics.update({f"train_f{k}": v for k, v in f1s.items()})

                bleus, f1s = text_loop_(ae, corrector, X_sub_test, Y_sub_test, idx_dataloader, rep=cfg.rep)
                val_metrics.update({f"test_b{k}": v for k, v in bleus.items()})
                val_metrics.update({f"test_f{k}": v for k, v in f1s.items()})
        
        val_metrics['epochs'] = i
        wandb.log(val_metrics)

        if (i + 1) % cfg.save_every == 0:
            torch.save(ae.state_dict(), save_dir + f'model_{i + 1}.pt')
    pbar.close()

if __name__ == "__main__":
    main()
