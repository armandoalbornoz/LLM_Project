import torch

def generate_text(model ,token_ids, max_new_tokens, context_size, temperature=1.0,  top_k = 50):
    """
    This function will autoregressively generate new tokens

    Args:
        model: the GPT model
        token_ids:  This is a tensor of shape  (batch, seq_len)
        max_new_tokens: This is the number of tokens we want to generate
        context_size: the maximum number of tokens the model can handle at once
    
    Returns:
        Tensor of shape (batch, seq_length + max_new_tokens)

    """
    model.eval()
    for _ in range(max_new_tokens):

        input_ids = token_ids[:, -context_size:] # crop to the last context size tokens
        logits = model(input_ids)  # (batch, tokens, dimension)
        next_logits = logits[:, -1, :] # grab the last token embedding of each batch (batch, dimension)

        # temperature
        next_logits = next_logits / max(temperature, 1e-8)

        # top-k

        if top_k > 0:
            top_vals, top_idx = torch.topk(next_logits, k=top_k, dim=-1)
            mask = torch.full_like(next_logits, float('-inf'))
            next_logits = mask.scatter(1, top_idx, top_vals)
            
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_ids = torch.cat([token_ids, next_token], dim=1) # adds the new token to the end of each batch
    
    return token_ids


