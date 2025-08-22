import tiktoken
import torch
from src.utils import generate_text

def token_ids_to_text(token_ids, tokenizer):
    """
    Transforms tensor of  token IDs  of size (batch_size, seq_length) into text 

    Args:
        token_ids: the token IDs to transform into text 
        tokenizer: A tokenizer to be used 
    """
    token_ids_flat = token_ids.squeeze(0)
    return tokenizer.decode(token_ids_flat.tolist())