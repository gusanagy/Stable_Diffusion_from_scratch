import torch
from torch import nn
from torch.nn import functional as F
import math

class selfAttention(nn.Module):

    def __init__(self, n_heads:int, d_embed:int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj_bias = nn.Liner(d_embed, 3 * d_embed, bias = in_proj_bias)
        self.out_proj_bias = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x (batch size, Seq_len, Dim)
        input_shape = x.shape
        batch_size, sequence_length, dim = input_shape
        intermin_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (batch size, Seq_len, Dim) -> (batch size, Seq_len, 3 * Dim) -> 3 tensors of shape (batch size, Seq_len, Dim)
        q, k, v = self.in_proj(x).chunk(3,dim=-1)

        # (batch_size,seq_len, Dim) -> (batch_size, seq_len, H, Dim / H) -> (batch_size, seq_len, H, Dim / H)
        q = q.view(intermin_shape).transpose(1,2)
        k = k.view(intermin_shape).transpose(1,2)
        v = v.view(intermin_shape).transpose(1,2)
        # (batch_size, seq_len, H, Dim / H) -> (batch_size, seq_len, seq_len,)
        weight = q @ k.transpose(-1,-2)

        if causal_mask:
            #mask where the upper triangle (above the principal diagonal) is made up of 1 
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head )

        weight = F.softmax(weight,dim=1)

        # (batch_size, H, seq_len, Seq_len) @ (Batch_size, H, Seq_len, Dim / H) -> (Batch_Size, H, Seq_len, Dim / H)
        output = weight @ v

        # (batch_seize, H, seq_len, Dim / H) -> (Batch_size, Seq_len, H,  Dim / H)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj_bias(output)

        # (batch_seize, seq_len, Dim)
        return output




