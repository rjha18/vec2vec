import nltk
import evaluate

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from utils.streaming_utils import process_batch

import matplotlib.pyplot as plt
import seaborn as sns


def generate_text(inverter, embeddings, max_seq_length=32):
    gen_kwargs = {
        "early_stopping": False,
        "num_beams": 1,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
        'min_length': 1,
        'max_length': max_seq_length,
    }
    regenerated = inverter.generate(
        inputs={
            "frozen_embeddings": embeddings,
        },
        generation_kwargs=gen_kwargs,
    )

    output_strings = inverter.tokenizer.batch_decode(
        regenerated, skip_special_tokens=True
    )
    return output_strings


def _calculate_token_f1(predictions, references):
    true_words = set(nltk.tokenize.word_tokenize(references))
    pred_words = set(nltk.tokenize.word_tokenize(predictions))

    TP = len(true_words & pred_words)
    FP = len(true_words) - len(true_words & pred_words)
    FN = len(pred_words) - len(true_words & pred_words)

    precision = (TP) / (TP + FP + 1e-20)
    recall = (TP) / (TP + FN + 1e-20)

    try:
        f1 = (2 * precision * recall) / (precision + recall + 1e-20)
    except ZeroDivisionError:
        f1 = 0.0
    return f1


def calculate_scores(score_flag, target_text, translation_text):
    if score_flag == 'bleu':
        bleu = evaluate.load("sacrebleu")
        score_func = lambda p, r: bleu.compute(predictions=[p], references=[r])['score']
    elif score_flag == 'f1':
        score_func = _calculate_token_f1
    else:
        raise ValueError(f"Unknown score_flag: {score_flag}")

    return np.mean([score_func(p, r) for p, r in zip(target_text, translation_text)])


def eval_batch(ins, recons, translations):
    recon_res = {}
    translation_res = {}
    for target_flag, emb in ins.items():
        emb = emb / emb.norm(dim=1, keepdim=True)
        in_distances = 1 - (emb @ emb.T)
        rec = recons[target_flag]
        rec = rec / rec.norm(dim=1, keepdim=True)
        rec_distances = 1 - (rec @ rec.T)
        recon_res[target_flag] = {
            "mse": F.mse_loss(emb, rec).item(),
            "cos": F.cosine_similarity(emb, rec).mean().item(),
            "std": rec.std(dim=0).mean().item(),
            "vsp": (in_distances - rec_distances).abs().mean().item()
        }
        translation_res[target_flag] = {}
        for flag, trans in translations[target_flag].items():
            trans = trans / trans.norm(dim=1, keepdim=True)
            out_distances = 1 - (trans @ trans.T)
            translation_res[target_flag][flag] = {
                "mse": F.mse_loss(emb, trans).item(),
                "cos": F.cosine_similarity(emb, trans).mean().item(),
                "std": trans.std(dim=0).mean().item(),
                "vsp": (in_distances - out_distances).abs().mean().item()
            }
    return recon_res, translation_res


def text_batch(ins, recons, translations, inverters, max_seq_length=32):
    recon_res = {}
    translation_res = {}

    for target_flag, inverter in inverters.items():
        gt = generate_text(inverter, ins[target_flag], max_seq_length=max_seq_length)

        rec = recons[target_flag]
        rec = rec / rec.norm(dim=1, keepdim=True)
        rec_text = generate_text(inverter, rec, max_seq_length=max_seq_length)
        recon_res[target_flag] = {
            "bleu": calculate_scores('bleu', gt, rec_text),
            "f1": calculate_scores('f1', gt, rec_text)
        }
        translation_res[target_flag] = {}
        for flag, trans in translations[target_flag].items():
            trans = trans / trans.norm(dim=1, keepdim=True)
            trans_text = generate_text(inverter, trans, max_seq_length=max_seq_length)
            translation_res[target_flag][flag] = {
                "bleu": calculate_scores('bleu', gt, trans_text),
                "f1": calculate_scores('f1', gt, trans_text)
            }
            print('gt:', gt[0])
            print('rec:', rec_text[0])
            print(f'trans ({flag} -> {target_flag}):', trans_text[0])
    return recon_res, translation_res


def merge_dicts(full, incremental):
    def recursive_merge(f, i):
        for key, val in i.items():
            if isinstance(val, dict):
                if key not in f or not isinstance(f[key], dict):
                    f[key] = {}
                recursive_merge(f[key], val)
            else:
                if key not in f:
                    f[key] = []
                f[key].append(val)
    
    recursive_merge(full, incremental)


def mean_dicts(full):
    def recursive_mean(f):
        for key, val in f.items():
            if isinstance(val, dict):
                recursive_mean(val)
            elif isinstance(val, list) and len(val) > 1:
                f[key] = np.mean(val)
            else:
                f[key] = val[0]
    
    recursive_mean(full)


def top_k_accuracy(sims, k=1):
    top_k_preds = np.argsort(sims, axis=1)[:, -k:]  # Get indices of top-k predictions
    correct = np.arange(sims.shape[0])[:, None]  # Ground truth indices
    return np.mean(np.any(top_k_preds == correct, axis=1))  # Check if correct label is in top-k


def get_avg_rank(sims: np.ndarray) -> float:
    ranks = (np.argsort(-sims) == np.arange(sims.shape[0])[:, None])
    return ranks.argmax(1).mean() + 1


def create_heatmap(translator, ins, tgt_emb, src_emb, top_k_size, heatmap_size=None, k=16) -> dict:
    res = {}
    ins = {k: v[:top_k_size] for k, v in ins.items()}
    trans = translator.translate_embeddings(ins[src_emb], src_emb, tgt_emb)
    ins_norm = F.normalize(ins[tgt_emb].cpu(), p=2, dim=1)
    trans_norm = F.normalize(trans.cpu(), p=2, dim=1)
    sims = (ins_norm @ trans_norm.T).numpy()
    res[f'{src_emb}_{tgt_emb}_top_1_acc'] = (sims.argmax(axis=1) == np.arange(sims.shape[0])).mean()
    res[f'{src_emb}_{tgt_emb}_top_{k}_acc'] = top_k_accuracy(sims, k)
    res[f"{src_emb}_{tgt_emb}_top_rank"] = get_avg_rank(sims)
    if heatmap_size is not None:
        sims = sims[:heatmap_size, :heatmap_size]
        sims_softmax = F.softmax(torch.tensor(sims) * 100, dim=1).numpy()
        res[f'{src_emb}_{tgt_emb}_heatmap_top_1_acc'] = top_k_accuracy(sims, 1)
        if heatmap_size > k:
            res[f'{src_emb}_{tgt_emb}_heatmap_top_{k}_acc'] = top_k_accuracy(sims, k)

        # plot sims
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(sims, vmin=0, vmax=1, cmap='coolwarm', ax=ax)
        ax.set_title('Heatmap of cosine similarities')
        ax.set_xlabel(f'Fake ({src_emb}->{tgt_emb})')
        ax.set_ylabel(f'Real ({tgt_emb})')
        plt.tight_layout()
        res[f'{src_emb}_{tgt_emb}_heatmap'] = fig 
        plt.close(fig)
    
        # plot sims w/ softmax
        fig, ax = plt.subplots(figsize=(8,7))
        sns.heatmap(sims_softmax, cmap='Purples', ax=ax)
        ax.set_title('Heatmap of cosine similarities (softmaxed)')
        ax.set_xlabel(f'Fake ({src_emb}->{tgt_emb})')
        ax.set_ylabel(f'Real ({tgt_emb})')
        plt.tight_layout()
        res[f'{src_emb}_{tgt_emb}_heatmap_softmax'] = fig 
        plt.close(fig)

    return res

def eval_loop_(
    cfg, translator, encoders, iter, inverters=None, pbar=None, device='cpu'
):
    recon_res = {}
    translation_res = {}
    heatmap_res = {}
    text_recon_res = {}
    text_translation_res = {}

    top_k_batches = cfg.top_k_batches if hasattr(cfg, 'top_k_batches') else 0
    text_batches = cfg.text_batches if hasattr(cfg, 'text_batches') else 0
    with torch.no_grad():
        n = 0
        for i, batch in enumerate(iter):
            ins = process_batch(cfg, batch, encoders, device)                
            n += cfg.val_bs
            recons, translations = translator(ins, include_reps=False)
            
            r_res, t_res = eval_batch(ins, recons, translations)
            merge_dicts(recon_res, r_res)
            merge_dicts(translation_res, t_res)
            if i < top_k_batches and hasattr(cfg, 'top_k_size') and hasattr(cfg, 'k') and cfg.top_k_size > 0:
                heatmap_size = cfg.heatmap_size if i == top_k_batches - 1 else None
                batch_res = create_heatmap(translator, ins, cfg.sup_emb, cfg.unsup_emb, cfg.top_k_size, heatmap_size, cfg.k)
                batch_res.update(create_heatmap(translator, ins, cfg.unsup_emb, cfg.sup_emb, cfg.top_k_size, heatmap_size, cfg.k))
                merge_dicts(heatmap_res, batch_res)
            if i < text_batches and inverters is not None:
                t_r_res, t_t_res = text_batch(ins, recons, translations, inverters)
                merge_dicts(text_recon_res, t_r_res)
                merge_dicts(text_translation_res, t_t_res)
            if pbar is not None:
                pbar.update(1)

        mean_dicts(recon_res)
        mean_dicts(translation_res)
        mean_dicts(heatmap_res)
        mean_dicts(text_recon_res)
        mean_dicts(text_translation_res)
        return recon_res, translation_res, heatmap_res, text_recon_res, text_translation_res


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_val_cos = 0

    def early_stop(self, val_cos):
        if val_cos > (self.max_val_cos + self.min_delta):
            self.max_val_cos = val_cos
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
