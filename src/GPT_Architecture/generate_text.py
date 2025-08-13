#generate one new token per iteration, for a total of max_new_tokens steps.
import torch
import torch.nn as nn

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is a (batch, n_tokens) array of indices in the current context.

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size :]
        with torch.no_grad(): # This tells PyTorch: "I'm not training — just doing inference — so don’t track gradients."
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # selects the logits for the last token position in the sequence
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
