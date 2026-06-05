import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self,d_in,d_out, context_size, dropout_rate, num_heads, qkv_bias):
        super().__init__()
        #d_out is the length of the embedding formed after multiplying weights with input embeddings
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads" 
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.W_Query = nn.Linear(d_in,d_out, bias=qkv_bias)
        self.W_Key = nn.Linear(d_in,d_out, bias=qkv_bias)
        self.W_Value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.num_heads = num_heads
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_size,context_size))
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,X):
        b, num_tokens, d_in = X.shape
        queries = self.W_Query(X)
        keys = self.W_Key(X)
        values = self.W_Value(X)

        # We implicitly split the matrix by adding a num_heads dimension. 
        # Then we unroll the last dim: 
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(                                             
            b, num_tokens, self.num_heads, self.head_dim                    
        )                                                                   

        #Transposes from shape (b, num_tokens, num_heads, head_dim) 
        # to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)


        atten_score = queries @ keys.transpose(2,3) # Computes dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        atten_score.masked_fill_(~mask_bool, -torch.inf)
        scaling_factor = atten_score.shape[-1] ** 0.5

        scaled_atten_score = atten_score / scaling_factor
        atten_weight = torch.softmax(scaled_atten_score, dim = -1)

        atten_weight = self.dropout(atten_weight)
        context_vectors = (atten_weight @ values).transpose(1,2)

        context_vectors = context_vectors.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vectors = self.out_proj(context_vectors)
        return context_vectors
        


