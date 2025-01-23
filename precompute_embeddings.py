from pathlib import Path
import os
import torch
import vec2text
from datasets import load_dataset
from utils.model_utils import load_encoder_tokenizer


TOKEN_LIMIT = 32

model_flags = ["gtr", "gte"]
device='cuda'
ds_flag = 'jxm/nq_corpus_dpr'
ds_folder = ds_flag.replace('/', '_')

home_dir = "/private/home/jxm/unsupervised_ae_translation/embeddings"
output_dir = os.path.join(home_dir, f'embeddings_v2_{TOKEN_LIMIT}/{ds_folder}/')
cache_dir = os.path.join(home_dir, f"data/{ds_folder}/")
dataset = load_dataset(ds_flag, cache_dir=cache_dir)

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

def transform(x, encoder, tokenizer, max_length=TOKEN_LIMIT, device='cpu'):
    embeddings = get_embeddings(x['text'], encoder, tokenizer, max_length, device)
    return {
        # 'label': x['label'],
        'text': x['text'],
        'text_embeddings': embeddings.cpu().numpy()
    }

for f in model_flags:
    encoder, tokenizer = load_encoder_tokenizer(f, device)
    T = lambda x: transform(x, encoder, tokenizer, max_length=TOKEN_LIMIT, device=device)
    dset = dataset.map(T, batched=True, batch_size=32, keep_in_memory=True)
    Path(output_dir + f).mkdir(parents=True, exist_ok=True)
    dset.save_to_disk(output_dir + f)
