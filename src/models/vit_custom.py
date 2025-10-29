import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, ViTForImageClassification

from .attention.base_attention import SoftmaxAttention
from .attention.entmax_attention import EntmaxAttention
from .attention.sparsemax_attention import SparsemaxAttention

class ViTCustomClassifier(nn.Module):
    def __init__(self, base_model_name, attention_type='softmax', num_labels=10):
        super().__init__()
        # 기본 ViT 설정
        self.config = ViTConfig.from_pretrained(base_model_name, num_labels=num_labels)
        self.vit    = ViTModel.from_pretrained(base_model_name, config=self.config)

        hidden_size = self.config.hidden_size
        num_heads   = self.config.num_attention_heads

        # 커스터마이즈 어텐션 블록
        if attention_type == 'entmax':
            self.custom_attn = EntmaxAttention(hidden_size, num_heads)
        elif attention_type == 'sparsemax':
            self.custom_attn = SparsemaxAttention(hidden_size, num_heads)
        else:
            self.custom_attn = SoftmaxAttention(hidden_size, num_heads)

        # 분류 헤드
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        # 기본 ViT 임베딩/패치 → 마지막 Hidden states
        vit_outputs = self.vit(pixel_values=pixel_values, output_hidden_states=False, output_attentions=False)
        sequence_output = vit_outputs.last_hidden_state  # shape: [B, num_patches+1, hidden_size]

        # 커스터마이즈 어텐션 적용 (CLS 토큰 포함 전체 시퀀스)
        attn_output, attn_weights = self.custom_attn(sequence_output)

        # CLS 토큰 위치는 인덱스 0
        cls_output = attn_output[:, 0, :]

        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits, 'attn_weights': attn_weights}
