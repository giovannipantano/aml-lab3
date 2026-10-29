import os
import torch
import matplotlib
matplotlib.use("Agg")  # backend senza GUI (funziona anche su Colab/server)
import matplotlib.pyplot as plt

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize(img):
    # Inverte la Normalize del dataloader per poter visualizzare l'immagine
    img = img.cpu() * STD + MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

def get_class_names(classes, data_dir="data/tiny-imagenet-200"):
    # Converte gli id WordNet (es. n01443537) in nomi leggibili usando words.txt
    words_file = os.path.join(data_dir, "words.txt")
    if not os.path.exists(words_file):
        return list(classes)
    wnid_to_name = {}
    with open(words_file, "r") as f:
        for line in f:
            wnid, name = line.strip().split("\t")
            wnid_to_name[wnid] = name.split(",")[0]
    return [wnid_to_name.get(c, c) for c in classes]

def _make_grid(images, titles, colors=None, ncols=4):
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            ax.imshow(denormalize(images[i]))
            ax.set_title(titles[i], fontsize=9, color=colors[i] if colors else "black")
    fig.tight_layout()
    return fig

def show_batch(loader, class_names, n=16, save_path=None):
    # Mostra n immagini del dataset con la relativa etichetta
    images, labels = next(iter(loader))
    n = min(n, images.size(0))
    titles = [class_names[l] for l in labels[:n]]
    fig = _make_grid(images[:n], titles)
    if save_path:
        fig.savefig(save_path)
    return fig

def show_predictions(model, loader, class_names, device, n=16, save_path=None):
    # Mostra predizione vs etichetta reale (verde = corretta, rosso = errata)
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:n], labels[:n]
    with torch.no_grad():
        preds = model(images.to(device)).argmax(1).cpu()
    titles = [f"P: {class_names[p]}\nT: {class_names[t]}" for p, t in zip(preds, labels)]
    colors = ["green" if p == t else "red" for p, t in zip(preds, labels)]
    fig = _make_grid(images, titles, colors)
    if save_path:
        fig.savefig(save_path)
    return fig

def plot_history(history, save_path=None):
    # Curve di loss e accuracy per train e validation
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
