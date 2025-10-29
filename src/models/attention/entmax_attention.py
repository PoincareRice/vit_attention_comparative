import torch
import torch.nn as nn
from entmax import entmax_bisect

class EntmaxAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, alpha=1.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.alpha     = alpha

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, N, C = x.size()
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1,2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = entmax_bisect(scores, alpha=self.alpha, dim=-1)
        attn_output = attn_weights @ v

        attn_output = attn_output.transpose(1,2).contiguous().view(B, N, C)
        return self.out(attn_output), attn_weights
