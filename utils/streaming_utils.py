import os

import datasets
import torch
import vec2text

from datasets import Features, Value, load_dataset
from beir.datasets.data_loader_hf import HFDataLoader


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"



def _prepare_retrieval_dataset(d: datasets.Dataset) -> datasets.Dataset:
    q_ds = d.remove_columns([c for c in d.column_names if c not in {'query', 'dataset'}]).rename_column('query', 'text')
    d_ds = d.remove_columns([c for c in d.column_names if c not in {'document', 'dataset'}]).rename_column('document', 'text')
    return datasets.concatenate_datasets([q_ds, d_ds])


def _load_retrieval_dataset() -> datasets.Dataset:
    d1 = datasets.load_dataset(
        "jxm/nomic_embed_supervised",
        num_proc=32,
        keep_in_memory=False,
    )["train"]
    d1 = _prepare_retrieval_dataset(d1)
    d2 = datasets.load_dataset(
        "jxm/nomic_embed_unsupervised",
        num_proc=32,
        keep_in_memory=False,
    )["train"]
    d2 = _prepare_retrieval_dataset(d2)
    return datasets.concatenate_datasets([d1, d2])


def load_streaming_embeddings(
        dataset_name: str,
        split_flag: str = "train",
        streaming: bool = False,
    ):
    if dataset_name == 'nq':
        dset = load_dataset("jxm/nq_corpus_dpr", split=split_flag, streaming=streaming)
    elif dataset_name == 'fineweb':
        dset = load_dataset("HuggingFaceFW/fineweb", streaming=streaming, num_proc=8, keep_in_memory=False)["train"]
    elif dataset_name == 'fineweb-medium':
        dset = load_dataset("HuggingFaceFW/fineweb", "sample-350BT", streaming=streaming, num_proc=8, keep_in_memory=False)["train"]
    elif dataset_name == "fineweb-tiny":
        dset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=streaming, num_proc=8)["train"]
    elif dataset_name == "nq-corpus":
        dset = load_dataset("BeIR/nq", "corpus", streaming=streaming, num_proc=8)["corpus"]
    elif dataset_name == "arguana-corpus":
        dset = load_dataset("BeIR/arguana", "corpus", streaming=streaming, num_proc=8)["corpus"]
    elif dataset_name == "arguana-queries":
        dset = load_dataset("BeIR/arguana", "queries", streaming=streaming, num_proc=8)["queries"]
    elif dataset_name == "fiqa-corpus":
        dset = load_dataset("BeIR/fiqa", "corpus", streaming=streaming, num_proc=8)["corpus"]
    elif dataset_name == "fiqa-queries":
        dset = load_dataset("BeIR/fiqa", "queries", streaming=streaming, num_proc=8)["queries"]
    elif dataset_name == "quora-corpus":
        dset = load_dataset("BeIR/quora", "corpus", streaming=streaming, num_proc=8)["corpus"]
    elif dataset_name == "quora-queries":
        dset = load_dataset("BeIR/quora", "queries", streaming=streaming, num_proc=8)["queries"]
    elif dataset_name == "trec-covid-corpus":
        dset = load_dataset("BeIR/trec-covid", "corpus", streaming=streaming, num_proc=8)["corpus"]
    elif dataset_name == "trec-covid-queries":
        dset = load_dataset("BeIR/trec-covid", "queries", streaming=streaming, num_proc=8)["queries"]
    elif dataset_name == "fever-corpus":
        dset = load_dataset("BeIR/fever", "corpus", streaming=streaming, num_proc=8)["corpus"]
    elif dataset_name == "fever-queries":
        dset = load_dataset("BeIR/fever", "queries", streaming=streaming, num_proc=8)["queries"]
    elif dataset_name == "scifact-corpus":
        dset = load_dataset("BeIR/scifact", "corpus", streaming=streaming, num_proc=8)["corpus"]
    elif dataset_name == "retrieval":
        dset = _load_retrieval_dataset()
    else:
        raise NotImplementedError()

    return dset.with_format("torch")


def get_embeddings(text_list,
                   encoder,
                   tokenizer,
                   max_length,
                   device):

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=max_length,
                       truncation=True,
                       padding="max_length").to(device)

    with torch.no_grad():
        model_output = encoder(**inputs)
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings

def embed(x, encoder, tokenizer, max_length=32, device='cpu'):
    embeddings = get_embeddings(x['text'], encoder, tokenizer, max_length, device)
    return {
        'text': x['text'],
        'text_embeddings': embeddings
    }

def forward_embedding_sentence_transformers(enc, features, normalize_embeddings: bool = True, mixed_precision: str = None):
    output_value  = "sentence_embedding"
    if mixed_precision is not None:
        if mixed_precision == 'bf16':
            enc_type = torch.bfloat16
        elif mixed_precision == 'fp16':
            enc_type = torch.float16
        else:
            raise ValueError(f"Unknown mixed precision flag {mixed_precision}")
    else:
        enc_type = torch.float32
    with torch.no_grad(), torch.autocast("cuda", dtype=enc_type):
        out_features = enc.forward(features)
    embeddings = out_features[output_value]
    embeddings = embeddings.detach()
    if normalize_embeddings:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.to(torch.float32)


def process_batch(cfg, batch, encoders, device='cpu'):
    ins = {}
    batch_embs = [k.replace("_input_ids", "") for k in batch.keys() if k.endswith("_input_ids")]
    for emb in batch_embs:
        encoders[emb].to(device)
        emb_inputs = { k.replace(f"{emb}_", ""): v for k, v in batch.items() if k.startswith(f"{emb}_") }
        ins[emb] = forward_embedding_sentence_transformers(
            encoders[emb], emb_inputs,
            normalize_embeddings=cfg.normalize_embeddings
        )
    return ins


class NanoBeirHFDataLoaderOverride(HFDataLoader):
    def _load_qrels(self, split):
        qrels_ds = load_dataset(
            self.hf_repo,
            "qrels"
        )["train"]
        qrels_ds = qrels_ds.add_column("score", [1] * len(qrels_ds))
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds


def load_beir_style_dataset(dataset: str):
    if 'nano' in dataset.lower():
        corpus, queries, qrels = NanoBeirHFDataLoaderOverride(
            hf_repo=f"zeta-alpha-ai/{dataset}",
            streaming=False,
            keep_in_memory=True
        ).load()
    else:
        corpus, queries, qrels = HFDataLoader(
            hf_repo=f"BeIR/{dataset.lower()}",
            streaming=False,
            keep_in_memory=False
        ).load(split="test")
    return corpus, queries, qrels