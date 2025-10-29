import os
import torch
import yaml
from tqdm import tqdm
from transformers import AutoImageProcessor
from src.data.load_dataset import get_dataloaders
from src.models.vit_custom import ViTCustomClassifier

def main():
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_dataloaders()

    model = ViTCustomClassifier(
        base_model_name=cfg['model']['base_model'],
        attention_type=cfg['model']['attention_type'],
        num_labels=cfg['model']['num_labels']
    ).to(device)

    image_processor = AutoImageProcessor.from_pretrained(
        cfg['model']['base_model'],
        do_rescale=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])

    exp_name = cfg['experiment']['exp_name']
    checkpoint_dir = os.path.join("experiments", cfg['experiment']['exp_name'], "checkpoints")
    results_dir = os.path.join("experiments", exp_name, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_losses, val_losses = [], []

    for epoch in range(cfg['training']['epochs']):
        # train
        model.train()
        total_train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}"):
            images = image_processor(images, return_tensors="pt").pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        total_val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Val]"):
                images = image_processor(images, return_tensors="pt").pixel_values.to(device)
                labels = labels.to(device)

                outputs = model(pixel_values=images, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]

                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = total_val_loss / len(test_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")


    save_path = os.path.join(checkpoint_dir, f"vit_{cfg['model']['attention_type']}_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved checkpoint: {save_path}")

    log_path = os.path.join(results_dir, "training_log.txt")
    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}\n")
        
    # 전체 학습 곡선 저장
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "loss_curve.png"))
    plt.close()
    print(f"📈 Saved training curve to: {os.path.join(results_dir, 'loss_curve.png')}")



if __name__ == "__main__":
    main()
