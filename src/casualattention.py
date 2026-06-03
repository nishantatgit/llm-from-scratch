import torch
class CasualAttention(torch.nn.Module):
    def __init__(self,in_dimension,out_dimension, context_size, qkv_bias=False):
        super().__init__()
        self.Q_Weight = torch.nn.Linear(in_dimension,out_dimension, bias = qkv_bias)
        self.K_Weight = torch.nn.Linear(in_dimension, out_dimension, bias = qkv_bias)
        self.V_Weight = torch.nn.Linear(in_dimension, out_dimension, bias = qkv_bias)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_size, context_size))
        )
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        queries = self.Q_Weight(x)
        keys = self.K_Weight(x)
        values = self.V_Weight(x)

        atten_score = queries @ keys.transpose(1, 2)
        atten_score = atten_score.masked_fill(self.mask == 0, -torch.inf)
        scaling_factor = atten_score.shape[-1] ** 0.5

        scaled_atten_score = atten_score / scaling_factor
        atten_weight = torch.softmax(scaled_atten_score, dim = -1)

        atten_weight = self.dropout(atten_weight)
        context_vectors = atten_weight @ values
        return context_vectors
        