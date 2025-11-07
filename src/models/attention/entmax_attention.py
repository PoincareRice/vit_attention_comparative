import torch
import torch.nn as nn
from entmax import entmax_bisect
from transformers.models.vit.modeling_vit import ViTSelfAttention

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

    def forward(self, x, head_mask=None):
        B, N, C = x.size()
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1,2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = entmax_bisect(scores, alpha=self.alpha, dim=-1)

        # head_mask가 있을 경우 적용 (optional)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
            
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1,2).contiguous().view(B, N, C)
        return self.out(attn_output), attn_weights
    

def replace_vit_attention_with_entmax(vit_model):
    hidden_size = vit_model.config.hidden_size
    num_heads   = vit_model.config.num_attention_heads
    for layer in vit_model.encoder.layer:
        if hasattr(layer.attention, "attention") and isinstance(layer.attention.attention, ViTSelfAttention):
            layer.attention.attention = EntmaxAttention(hidden_size, num_heads)
    return vit_model