import torch
import torch.nn as nn
import numpy as np
from math import sqrt

from utils import TriangularCausalMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape # H = self.n_heads
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) 
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # fw = open("attention_matrix.pkl", "wb")
        # pickle.dump(A, fw)
        # fw.close()
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, key_padding_mask):
        B, L, _ = queries.shape #torch.Size([128, 200, 50])  batch_size * seq_len * hidden_size
        _, S, _ = keys.shape #torch.Size([128, 200, 50])  batch_size * seq_len * hidden_size
        H = self.n_heads

        # queries = self.query_projection(queries).view(B, L, H, -1) #torch.Size([128, 200, 1, 50])
        # keys = self.key_projection(keys).view(B, S, H, -1) #torch.Size([128, 200, 1, 50])
        # values = self.value_projection(values).view(B, S, H, -1) #torch.Size([128, 200, 1, 50])

        # 增加对padding位置的处理使其变回0，防止对M的计算造成影响
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        timeline_mask = key_padding_mask.unsqueeze(-1)
        queries *= ~timeline_mask
        keys *= ~timeline_mask
        values *= ~timeline_mask

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, L, H, -1)
        values = values.view(B, L, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ) # torch.Size([32, 96, 8, 64])
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1) #

        return self.out_projection(out), attn
