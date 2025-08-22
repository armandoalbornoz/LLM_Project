import torch
import torch.nn as nn
from src.MultiHeadAttention import MultiHeadAttention


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

        self.transformer_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])] 
        )

        self.final_normalization = NormalizationLayer(config["emb_dim"])
        
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, input_IDs):
        batch_size, seq_len = input_IDs.shape
        token_embeds = self.token_emb(input_IDs)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=input_IDs.device))

        x = token_embeds + pos_embeds
        x = self.dropout(x)
        x = self.transformer_block(x)
        x = self.final_normalization(x)
        y = self.out_head(x)
        return y
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
                d_in=config["emb_dim"],
                d_out=config["emb_dim"],
                context_length=config["context_length"],
                num_heads=config["n_heads"],
                dropout=config["drop_rate"],
                qkv_bias = config["qkv_bias"])
        
        self.ff = FeedForward(config=config)
        self.norm1 = NormalizationLayer(config["emb_dim"])
        self.norm2 = NormalizationLayer(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    

class NormalizationLayer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # we want this to ensure we don't divide by zero
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.epsilon)
        return self.scale * norm_x + self.shift
