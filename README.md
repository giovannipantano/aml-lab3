# Lab - Setup a project from scratch

Image classification on **Tiny ImageNet** (200 classes, 64x64) with a custom CNN, tracked with **Weights & Biases**.

## Structure

```
checkpoints/   saved models (best_model.pth) - not pushed
data/          the actual dataset - not pushed
dataset/       dataloader / dataset classes
models/        architectures (CustomNet)
utils/         dataset download + visualization helpers
outputs/       generated plots - not pushed
train.py       training logic
eval.py        evaluation logic
```

## Local usage

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python utils/download_dataset.py # downloads Tiny ImageNet into data/
wandb login
python train.py --epochs 10 --batch_size 64 --lr 0.001
python eval.py
```

## Colab

```python
!git clone https://github.com/<your-username>/aml-lab3.git
%cd aml-lab3
!pip install -r requirements.txt
!python utils/download_dataset.py
import wandb; wandb.login()
!python train.py --epochs 10
!python eval.py
```

## Visualization

- `outputs/samples.png`: a batch of training images with their labels
- `outputs/training_curves.png`: train/val loss and accuracy per epoch
- `outputs/predictions.png`, `outputs/eval_predictions.png`: model predictions (green = correct, red = wrong)

The same figures, plus the per-epoch metrics, are logged to the W&B project `aml-lab3`.
