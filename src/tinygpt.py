import torch
import torch.nn as nn
import sys
sys.path.append("..")
from .multiheadattention import  MultiheadAttention 
class GPTModel(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        #define the token loopup table
        self.token_embd = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        #define positional embedding lookup table 
        self.pos_embd = nn.Embedding(cfg["context_length"],cfg["emb_dim"])

        # define the dropout 
        self.drop_embd = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["no_of_layers"])]
        )

        self.final_norms = LayerNorm(cfg["emb_dim"])

        # number of nodes in the out head needs to be of the size of vocab so that the 
        # logits are of the same number of the size of vocab referring to the probability of each word
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self,in_idx):
        # for example if the have 2 sequences of length 3 we will have a list consisting of 2 lists of 3 token ids
        # [[12,34,534], [324, 21,189]]

        batch_size, seq_len = in_idx.shape
        token_embd = self.token_embd(in_idx)
        pos_embd = self.pos_embd(torch.arange(seq_len, device=in_idx.device))

        x = token_embd + pos_embd
        x = self.drop_embd(x)
        x = self.trf_blocks(x)
        x = self.final_norms(x)
        logits = self.out_head(x)

        return logits

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiheadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_size = cfg["context_length"],
            dropout_rate=cfg["drop_rate"],
            num_heads = cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self,X):
        shortcut = X
        X = self.norm1(X)
        X = self.att(X)
        X = self.drop_shortcut(X)
        X = X + shortcut

        shortcut = X
        X = self.norm2(X)
        X = self.ff(X)
        X = self.drop_shortcut(X)
        X = X + shortcut
        return X
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean = x.mean(dim = - 1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"],cfg["emb_dim"])
        )

    def forward(self,X):
        return self.layers(X)
    

