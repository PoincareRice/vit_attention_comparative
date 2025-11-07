import torch
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTSelfAttention

def sparsemax(logits, dim=-1):
    z = logits - logits.max(dim=dim, keepdim=True)[0]
    sorted_z, _ = torch.sort(z, descending=True, dim=dim)
    cssv = torch.cumsum(sorted_z, dim=dim) - 1
    ind = torch.arange(1, logits.size(dim)+1, device=logits.device).view((1,)*(logits.dim()-1)+(-1,))
    cond = sorted_z > cssv / ind
    k = cond.sum(dim=dim, keepdim=True)
    tau = (cssv.gather(dim, k-1) / k)
    output = torch.clamp(z - tau, min=0)
    return output

class SparsemaxAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads

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
        attn_weights = sparsemax(scores, dim=-1)

                # head_mask가 있을 경우 적용 (optional)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1,2).contiguous().view(B, N, C)
        return self.out(attn_output), attn_weights

def replace_vit_attention_with_sparsemax(vit_model):
    hidden_size = vit_model.config.hidden_size
    num_heads   = vit_model.config.num_attention_heads
    for layer in vit_model.encoder.layer:
        if hasattr(layer.attention, "attention") and isinstance(layer.attention.attention, ViTSelfAttention):
            layer.attention.self = SparsemaxAttention(hidden_size, num_heads)
    return vit_model