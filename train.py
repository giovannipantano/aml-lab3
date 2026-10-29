import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from models.customnet import CustomNet
from dataset.dataloader import get_dataloaders
from utils.visualization import get_class_names, show_batch, show_predictions, plot_history

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    return running_loss / total, correct / total

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/tiny-imagenet-200")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()

    # Inizializzazione WandB
    run = wandb.init(
        project="aml-lab3",
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "architecture": "CustomNet",
            "dataset": "TinyImageNet"
        }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del device: {device}")

    # DataLoaders
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=config.batch_size,
        num_workers=args.num_workers
    )
    class_names = get_class_names(train_loader.dataset.classes, args.data_dir)

    # Visualizzazione di alcuni esempi del dataset
    os.makedirs("outputs", exist_ok=True)
    fig = show_batch(train_loader, class_names, save_path="outputs/samples.png")
    wandb.log({"samples": wandb.Image(fig)})

    # Modello, Loss, Ottimizzatore
    model = CustomNet(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Logging metriche su WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Salva il checkpoint del modello migliore
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

    # Curve di training e predizioni del modello finale
    fig = plot_history(history, save_path="outputs/training_curves.png")
    wandb.log({"training_curves": wandb.Image(fig)})
    fig = show_predictions(model, val_loader, class_names, device, save_path="outputs/predictions.png")
    wandb.log({"predictions": wandb.Image(fig)})

    wandb.finish()

if __name__ == "__main__":
    main()