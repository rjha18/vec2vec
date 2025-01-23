from sentence_transformers import SentenceTransformer
import torch


MODEL_PATH = '/private/home/jxm/supervised_translation/model_weights/'
# MODEL_PATH = "/home/rishi/code/data/model_weights/"

HF_FLAGS = {
    'gtr': 'sentence-transformers/gtr-t5-base',
    'gte': 'thenlper/gte-base',
    'gist': 'avsolatorio/GIST-Embedding-v0',
    'stella': 'infgrad/stella-base-en-v2',
    'sentence-t5': 'sentence-transformers/sentence-t5-base',
    'e5': 'intfloat/e5-base-v2',
    'ember': 'llmrails/ember-v1',
    'snowflake': 'Snowflake/snowflake-arctic-embed-m-long',
    'sbert': 'sentence-transformers/all-MiniLM-L12-v2',
    'clip': 'sentence-transformers/clip-ViT-B-32',
    'jina': 'jinaai/jina-embeddings-v2-base-en',
    'bert-nli': 'sentence-transformers/bert-base-nli-mean-tokens',
    'dpr': 'sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base',
}


def load_encoder(model_flag, device: str = 'cpu', max_seq_length: int = 32, mixed_precision: bool = False):
    if model_flag in HF_FLAGS:
        f = HF_FLAGS[model_flag]
    else:
        f = model_flag

    model_kwargs = {}
    if mixed_precision:
        model_kwargs['torch_dtype'] = torch.bfloat16
    
    encoder = SentenceTransformer(f, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
    encoder.max_seq_length = max_seq_length
    return encoder.eval()


def get_sentence_embedding_dimension(encoder):
    dim = encoder.get_sentence_embedding_dimension()
    if dim is not None:
        return dim

    # special handling for CLIP models
    dim = encoder[0].model.text_model.config.hidden_size

    return dim