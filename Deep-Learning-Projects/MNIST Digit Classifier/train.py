import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from model import DigitClassifier
from evaluate import evaluate, full_evaluation, plot_training_curves

NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=custom_transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=custom_transform)

# num_workers=0 here (vs. 2 in the notebook) — multiprocessing workers don't play well
# when this is invoked directly as a script on macOS/spawn platforms
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = DigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_accuracies = []

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mnist-classifier")

with mlflow.start_run(run_name="3layer_feedforward"):
    mlflow.log_params({
        "architecture": "784-256-128-10",
        "optimizer": "Adam",
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "activation": "ReLU",
        "loss": "CrossEntropyLoss"
    })

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_acc = evaluate(model, test_loader, device)

        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)

        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        print(f"Epoch : {epoch+1:02d}/{NUM_EPOCHS:02d} | Training Loss = {avg_loss:.4f} | val_acc = {val_acc:.4f}")

    mlflow.pytorch.log_model(model, "model")

    test_accuracy = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")

    full_evaluation(model, test_loader, device)
    plot_training_curves(train_losses, val_accuracies)

    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_artifact("reports/figures/loss_curve.png")
    mlflow.log_artifact("reports/figures/accuracy_curve.png")
    mlflow.log_artifact("reports/figures/confusion_matrix.png")

    torch.save(model.state_dict(), "model_weights.pt")
