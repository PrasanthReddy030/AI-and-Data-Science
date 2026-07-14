# MNIST Digit Classifier

A fully connected feedforward neural network for classifying handwritten digits (0-9), built from scratch in pure PyTorch — no Trainer API, no Lightning. Every part of the pipeline (data loading, model, training loop, evaluation) is written by hand to understand what's happening mechanically at each step.

## Architecture

```
Input: 784 (28x28 flattened)
  -> Linear(784, 256) -> ReLU
  -> Linear(256, 128) -> ReLU
  -> Linear(128, 10)   <- raw logits, no softmax
Output: 10 class logits
```

## Training Setup

| Setting | Value |
|---|---|
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Loss function | CrossEntropyLoss |
| Device | CPU |

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 97.80% |
| Parameters | 235,146 |
| Target | >= 97% |

### Loss Curve

![Loss Curve](reports/figures/loss_curve.png)

### Validation Accuracy Curve

![Accuracy Curve](reports/figures/accuracy_curve.png)

### Confusion Matrix

![Confusion Matrix](reports/figures/confusion_matrix.png)

**Key observation:** the model's most common mistake is misreading 7s as 9s (19 cases), closely followed by 4s and 5s also read as 9s (18 cases each) — 9 seems to be the "catch-all" wrong guess when a digit's top loop or stroke is ambiguous. 5s are also confused with 3s in 12 cases, likely from similar curved strokes on the right side of the digit. Every other pair stays in single digits, so there's no systemic weakness beyond these visually similar shapes. (Note: exact counts shift a few points between training runs due to random init/shuffling — the pattern of which digits confuse each other is the stable part.)

## Notes

- `Normalize((0.1307,), (0.3081,))` uses the precomputed mean/std of MNIST's training set pixel values (scaled to [0,1] by `ToTensor()`), so the network sees roughly zero-centered, unit-variance inputs.
- The test set doubles as the validation set for per-epoch monitoring, since `torchvision.datasets.MNIST` only ships a train/test split. No decisions (early stopping, hyperparameter tuning) were made based on per-epoch test performance, so this doesn't bias the final reported accuracy — but a dedicated held-out validation split would be the more rigorous setup for a non-practice project.

## How to Run

```bash
pip install -r requirements.txt
python train.py
```

`train.py` runs the full pipeline end to end — training, evaluation, and figure generation. `evaluate.py` isn't a standalone entry point; it's the module `train.py` imports for the accuracy check, classification report, and plots.

## MLflow

Training runs (params, per-epoch metrics, and final test accuracy) are tracked in MLflow using a local SQLite backend. View them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
