import torch
import torch.nn.functional as F


def rec_loss_fn(ins, recons, logger, prefix=""):
    assert ins.keys() == recons.keys()
    loss = None
    for flag, emb in ins.items():
        mse_loss = ((recons[flag] - emb) ** 2).sum(dim=1).sqrt().mean()
        logger.logkv(f"{prefix}{flag}_recons", mse_loss)
        if loss is None:
            loss = mse_loss
        else:
            loss += mse_loss
    return loss / len(ins)


def uni_loss_fn(emb, trans, src_emb, tgt_emb, logger):
    trans_loss = ((emb - trans) ** 2).sum(dim=1).sqrt().mean()
    logger.logkv(f"{src_emb}_{tgt_emb}_cos", F.cosine_similarity(emb, trans, dim=1).mean())
    return trans_loss

def trans_loss_fn(ins, translations, logger, prefix=""):
    assert ins.keys() == translations.keys()
    loss = None
    for target_flag, emb in ins.items():
        for flag, trans in translations[target_flag].items():
            mse_loss = ((emb - trans) ** 2).sum(dim=1).sqrt().mean()
            logger.logkv(f"{prefix}{flag}_{target_flag}_trans", mse_loss)
            
        if loss is None:
            loss = mse_loss
        else:
            loss += mse_loss
            logger.logkv(f"{prefix}{flag}_{target_flag}_cos", F.cosine_similarity(emb, trans, dim=1).mean())
    return (loss / len(ins))


def vsp_loss_fn(ins, translations, logger):
    assert ins.keys() == translations.keys()
    loss = None
    # TODO: Abstract this to work properly with non-unit vectors.
    for target_flag in ins.keys():
        in_distances = 1 - (ins[target_flag] @ ins[target_flag].T)
        for flag in translations[target_flag].keys():
            out_distances = 1 - (translations[target_flag][flag] @ translations[target_flag][flag].T)
            vsp_loss = (in_distances - out_distances).abs().mean()
            logger.logkv(f"{flag}_{target_flag}_vsp", vsp_loss)
            
        if loss is None:
            loss = vsp_loss
        else:
            loss += vsp_loss
    return (loss / len(ins))


def get_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # Calculate the 2-norm of the gradients
            total_norm += param_norm.detach() ** 2
    total_norm = total_norm ** (1. / 2)  # Take the square root to get the total norm
    return total_norm