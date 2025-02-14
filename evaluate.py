import torch
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm

def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    mode: str,
    wandb):
    """
    Evaluates the model on a given dataset.

    Args:
        model: The neural network model.
        dataloader: The DataLoader for validation or training evaluation.
        criterion: The loss function.
        device: The device to run the evaluation on (CPU/GPU).
        epoch: The current epoch number.
        mode: "valid" for validation, "train" for training evaluation.
        wandb: The Weights & Biases object for logging.

    Returns:
        A dictionary with loss, accuracy, precision, recall, and F1-score.
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0  # Initialize loss accumulator

    # Initialize metrics
    metric_acc = Accuracy(task="multiclass", num_classes=19).to(device)
    metric_prec = Precision(task="multiclass", num_classes=19, average="macro").to(device)
    metric_rec = Recall(task="multiclass", num_classes=19, average="macro").to(device)
    metric_f1 = F1Score(task="multiclass", num_classes=19, average="macro").to(device)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'{mode.capitalize()} Evaluation', unit='batch')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            metric_acc.update(preds, labels)
            metric_prec.update(preds, labels)
            metric_rec.update(preds, labels)
            metric_f1.update(preds, labels)

            progress_bar.set_postfix({f'{mode} Loss': running_loss / len(progress_bar)})

    # Compute final metrics
    loss_avg = running_loss / len(dataloader)
    accuracy = metric_acc.compute().item()
    precision = metric_prec.compute().item()
    recall = metric_rec.compute().item()
    f1 = metric_f1.compute().item()

    # Log metrics to W&B
    wandb.log({
        "epoch": epoch + 1,
        f"{mode}_loss": loss_avg,
        f"{mode}_acc": accuracy,
        f"{mode}_precision": precision,
        f"{mode}_recall": recall,
        f"{mode}_f1": f1
    })

    # Reset metric states for next evaluation
    metric_acc.reset()
    metric_prec.reset()
    metric_rec.reset()
    metric_f1.reset()

    return {
        "loss": loss_avg,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
