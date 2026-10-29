import os
import torch
import torch.nn as nn
from models.customnet import CustomNet
from dataset.dataloader import get_dataloaders
from utils.visualization import get_class_names, show_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(data_dir="data/tiny-imagenet-200", batch_size=64)

    model = CustomNet(num_classes=200).to(device)
    checkpoint_path = "checkpoints/best_model.pth"

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Pesi caricati con successo da {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} non trovato. Valutazione su pesi non addestrati.")

    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    final_loss = running_loss / total
    final_acc = correct / total
    print(f"\n--- RISULTATI VALUTAZIONE FINALE ---")
    print(f"Loss: {final_loss:.4f}")
    print(f"Accuracy: {final_acc * 100:.2f}%")

    # Salva una griglia con alcune predizioni del modello
    os.makedirs("outputs", exist_ok=True)
    class_names = get_class_names(val_loader.dataset.classes)
    show_predictions(model, val_loader, class_names, device, save_path="outputs/eval_predictions.png")
    print("Predizioni salvate in outputs/eval_predictions.png")

if __name__ == "__main__":
    main()