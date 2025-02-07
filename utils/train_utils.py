import torch
import torch.nn.functional as F


rmse = lambda x, y: ((x - y) ** 2).sum(dim=1).sqrt().mean()

def rec_loss_fn(ins, recons, logger, prefix=""):
    assert ins.keys() == recons.keys()
    loss = None
    for flag, emb in ins.items():
        recons_loss = 1 - F.cosine_similarity(emb, recons[flag], dim=1).mean()
        logger.logkv(f"{prefix}{flag}_recons_rmse", rmse(emb, recons[flag]))
        logger.logkv(f"{prefix}{flag}_recons_cos", recons_loss)
        if loss is None:
            loss = recons_loss
        else:
            loss += recons_loss
    return loss / len(ins)


def uni_loss_fn(emb, trans, src_emb, tgt_emb, logger):
    uni_loss = 1 - F.cosine_similarity(emb, trans, dim=1).mean()
    logger.logkv(f"{src_emb}_{tgt_emb}_uni_rmse", rmse(emb, trans))
    logger.logkv(f"{src_emb}_{tgt_emb}_uni_cos", uni_loss)
    return uni_loss


def trans_loss_fn(ins, translations, logger, prefix=""):
    assert ins.keys() == translations.keys()
    loss = None
    for target_flag, emb in ins.items():
        for flag, trans in translations[target_flag].items():
            trans_loss = 1 - F.cosine_similarity(emb, trans, dim=1).mean()
            logger.logkv(f"{prefix}{flag}_{target_flag}_trans_rmse", rmse(emb, trans))
            logger.logkv(f"{prefix}{flag}_{target_flag}_trans_cos", trans_loss)
            
            if loss is None:
                loss = trans_loss
            else:
                loss += trans_loss

    return (loss / len(ins))


def contrastive_loss_fn(ins, translations, logger) -> torch.Tensor:
# TODO: Think about this + test.
    loss = None
    EPS = 1e-10
    count = 0
    for out_name in ins.keys():
        for in_name in translations[out_name].keys():
            B = ins[out_name].detach()
            B = B / (B.norm(dim=1, keepdim=True) + EPS)
            in_sims = B @ B.T
            A = translations[out_name][in_name]
            A = A / (A.norm(dim=1, keepdim=True) + EPS)
            out_sims_reflected = A @ B.T
            contrastive_loss = torch.nn.functional.cross_entropy(
                out_sims_reflected * 50,
                torch.arange(in_sims.shape[0], device=in_sims.device)
            )
            if logger is not None:
                logger.logkv(f"{in_name}_{out_name}_contrastive", contrastive_loss)

            if loss is None:
                loss = contrastive_loss
            else:
                loss += contrastive_loss
            count += 1
    return loss / count

def vsp_loss_fn(ins, translations, logger) -> torch.Tensor:
    loss = None
    EPS = 1e-10
    count = 0
    for out_name in ins.keys():
        for in_name in translations[out_name].keys():
            B = ins[out_name].detach()
            B = B / (B.norm(dim=1, keepdim=True) + EPS)
            in_sims = B @ B.T
            A = translations[out_name][in_name]
            A = A / (A.norm(dim=1, keepdim=True) + EPS)
            out_sims = A @ A.T
            out_sims_reflected = A @ B.T
            vsp_loss = (
                  (in_sims - out_sims).abs().mean()
                + (in_sims - out_sims_reflected).abs().mean()
            )
            if logger is not None:
                logger.logkv(f"{in_name}_{out_name}_vsp", vsp_loss)

            if loss is None:
                loss = vsp_loss
            else:
                loss += vsp_loss
            count += 1
    return loss / count


def get_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # Calculate the 2-norm of the gradients
            total_norm += param_norm.detach() ** 2
    total_norm = total_norm ** (1. / 2)  # Take the square root to get the total norm
    return total_norm