import transformers

def get_tokenizer_max_length(tokenizer: transformers.PreTrainedTokenizer) -> int:
    if hasattr(tokenizer, 'tokenizer'):
        return tokenizer.tokenizer.model_max_length
    else:
        return tokenizer.model_max_length
