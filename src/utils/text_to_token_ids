import tiktoken
import torch
from src.utils import generate_text

def text_to_token_ids(text, tokenizer):
    """
    Transforms text into a tensor of  token IDs  of size (batch_size, seq_length) 

    Args:
        text: the text to be tokenized
        tokenizer: A tokenizer to be used 
    """
    token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0) # unsqueeze will add batch dimension
    return token_ids_tensor