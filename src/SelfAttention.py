import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, in_dimension, out_dimension, qkv_bias=False):
        super().__init__()
        self.W_Query = torch.nn.Linear(in_dimension, out_dimension, bias=qkv_bias)
        self.W_Key = torch.nn.Linear(in_dimension, out_dimension, bias=qkv_bias)
        self.W_Value = torch.nn.Linear(in_dimension, out_dimension, bias=qkv_bias)

    def forward(self,X):
        queries = self.W_Query(X)
        keys = self.W_Key(X)
        values = self.W_Value(X)

        attn_scores = queries @ keys.T
        scaling_factor = keys.shape[-1] ** 0.5
        attn_socres_scaled = attn_scores / scaling_factor

        attn_weights = torch.softmax(attn_socres_scaled, dim = -1)
        context_vector = attn_weights @ values

        return context_vector