import os
import numpy as np
from datasets import load_from_disk

home_dir = "/private/home/jxm/supervised_translation"
# home_dir = "/private/home/jxm/unsupervised_ae_translation/embeddings"
EMBEDDING_DIR = os.path.join(home_dir, 'embeddings_v2{}')

def normalize_example(ex):
    ex["text_embeddings"] = (ex["text_embeddings"].T / ex["text_embeddings"].norm(p=2, dim=1)).T
    return ex

def load_embeddings(
        dataset_name: str,
        embedding_flag: str,
        split_flag: str = "train",
        max_seq_length: int = 128,
        device: str = 'cpu'
    ):
    embedding_dir = EMBEDDING_DIR.format(f'_{max_seq_length}' if max_seq_length != 128 else '')
    if dataset_name == 'emotion':
        dset = load_from_disk(f'{embedding_dir}/emotion/{embedding_flag}')[split_flag]
    elif dataset_name == 'quora':
        dset = load_from_disk(f'{embedding_dir}/BeIR_quora/{embedding_flag}')['corpus']
    elif dataset_name == 'nq':
        dset = load_from_disk(f'{embedding_dir}/jxm_nq_corpus_dpr/{embedding_flag}')[split_flag]
    else:
        raise NotImplementedError()

    dset.set_format('torch', columns=['text', 'text_embeddings'], device=device)
    return dset


def load_and_process_embeddings(
    dataset_name: str,
    embedding_flag: str,
    N_SEEDS: np.ndarray,
    SEED: int = 0,
    normalize: bool = False,
    split_flag: str = "train",
    max_seq_length: int = 128,
    keep_in_memory: bool = True,
    device: str = 'cuda',
):
    dset = load_embeddings(dataset_name, embedding_flag, split_flag, max_seq_length, device)
    train_mask = np.array([0] * (len(dset) - N_SEEDS) + [1] * N_SEEDS, dtype=bool)
    np.random.seed(SEED)
    np.random.shuffle(train_mask)
    idxs = np.where(train_mask)[0]

    dset = dset.select(idxs)
    if normalize:
        dset = dset.map(normalize_example, batched=True, keep_in_memory=keep_in_memory)
    return dset, train_mask

def load_and_process_embeddings_from_idxs(
    dataset_name: str,
    embedding_flag: str,
    idxs: np.ndarray,
    normalize: bool = False,
    split_flag: str = "train",
    max_seq_length: int = 128,
    keep_in_memory: bool = True,
    device: str = 'cuda', 
):
    dset = load_embeddings(dataset_name, embedding_flag, split_flag, max_seq_length, device)
    dset = dset.select(idxs)
    if normalize:
        dset = dset.map(normalize_example, batched=True, keep_in_memory=keep_in_memory)
    return dset
