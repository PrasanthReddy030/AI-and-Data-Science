import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate(model, loader, device):
    """Cheap accuracy check, called after every epoch during training."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            predicted = torch.argmax(predictions, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


def full_evaluation(model, loader, device):
    """Detailed post-training report: per-class metrics + confusion matrix.

    Only meant to run once on the final model, not per-epoch — sklearn's
    classification_report and the heatmap render are too slow for that.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())  # stays on CPU, never touches the model

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title("MNIST Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.show()


def plot_training_curves(train_losses, val_accuracies,
                          loss_path="reports/figures/loss_curve.png",
                          acc_path="reports/figures/accuracy_curve.png"):
    """Save loss/accuracy curves as separate PNGs (kept apart since they use different y-scales)."""
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', markersize=4)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.show()
    print(f"Saved to {loss_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(val_accuracies)+1), [a*100 for a in val_accuracies], 'g-o', markersize=4)
    plt.axhline(y=97, color='r', linestyle='--', label='97% target')  # rough bar for this architecture, not a hard requirement
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.show()
    print(f"Saved to {acc_path}")

