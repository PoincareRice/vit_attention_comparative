import torch
import yaml
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor
from src.models.vit_custom import ViTCustomClassifier
from src.data.load_dataset import get_dataloaders

def evaluate_and_visualize():
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    attention_type = cfg['experiment']['attention_type']
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')

    model = ViTCustomClassifier(
        base_model_name=cfg['model']['base_model'],
        attention_type=attention_type,
        num_labels=cfg['model']['num_labels']
    ).to(device)

    checkpoint_path = cfg['experiment']['checkpoint_file']
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    model.eval()

    # 데이터 로더
    _, test_loader = get_dataloaders()
    image_processor = AutoImageProcessor.from_pretrained(
        cfg['model']['base_model'],
        do_rescale=False,
        use_fast=True
    )

    # 배치 하나만 테스트
    images, labels = next(iter(test_loader))
    images_proc = image_processor(images, return_tensors="pt").pixel_values.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=images_proc, labels=labels)
        logits      = outputs['logits']
        attn_weights= outputs['attn_weights']

    preds = torch.argmax(logits, dim=-1)
    acc   = (preds == labels).float().mean()
    print(f"Accuracy on batch: {acc:.4f}")

    # 어텐션 가중치 시각화 — 첫 배치의 첫 헤드
    attn_map = attn_weights[0,0]  # shape: [seq_len, seq_len]
    plt.imshow(attn_map.cpu().numpy(), cmap='viridis')
    plt.title(f"Attention map — {attention_type}")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    evaluate_and_visualize()