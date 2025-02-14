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
    Evaluates the model on a given dataset, computing both overall and per-class metrics.

    Args:
        model: The neural network model.
        dataloader: The DataLoader for validation or training evaluation.
        criterion: The loss function.
        device: The device to run the evaluation on (CPU/GPU).
        epoch: The current epoch number.
        mode: "valid" for validation, "train" for training evaluation.
        wandb: The Weights & Biases object for logging.

    Returns:
        A dictionary with loss, overall accuracy, precision, recall, and F1-score, 
        along with per-class versions of these metrics.
    """
    model.eval()
    running_loss = 0.0

    # Define number of classes
    num_classes = 19

    # Initialize overall and per-class metrics
    metric_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro").to(device)
    metric_prec = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
    metric_rec = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
    metric_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    metric_acc_per_class = Accuracy(task="multiclass", num_classes=num_classes, average="none").to(device)
    metric_prec_per_class = Precision(task="multiclass", num_classes=num_classes, average=None).to(device)
    metric_rec_per_class = Recall(task="multiclass", num_classes=num_classes, average=None).to(device)
    metric_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None).to(device)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'{mode.capitalize()} Evaluation', unit='batch')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Update both overall and per-class metrics
            metric_acc.update(preds, labels)
            metric_prec.update(preds, labels)
            metric_rec.update(preds, labels)
            metric_f1.update(preds, labels)

            metric_acc_per_class.update(preds, labels)
            metric_prec_per_class.update(preds, labels)
            metric_rec_per_class.update(preds, labels)
            metric_f1_per_class.update(preds, labels)

            progress_bar.set_postfix({f'{mode} Loss': running_loss / len(progress_bar)})

    # Compute final metrics
    loss_avg = running_loss / len(dataloader)
    
    accuracy = metric_acc.compute().item()
    precision = metric_prec.compute().item()
    recall = metric_rec.compute().item()
    f1 = metric_f1.compute().item()

    accuracy_per_class = metric_acc_per_class.compute().tolist()
    precision_per_class = metric_prec_per_class.compute().tolist()
    recall_per_class = metric_rec_per_class.compute().tolist()
    f1_per_class = metric_f1_per_class.compute().tolist()

    # Log metrics to W&B
    metrics_dict = {
        "epoch": epoch + 1,
        f"{mode}_loss": loss_avg,
        f"{mode}_accuracy": accuracy,
        f"{mode}_precision": precision,
        f"{mode}_recall": recall,
        f"{mode}_f1": f1
    }

    # Add per-class metrics
    for i in range(num_classes):
        metrics_dict[f"{mode}_accuracy_class_{i}"] = accuracy_per_class[i]
        metrics_dict[f"{mode}_precision_class_{i}"] = precision_per_class[i]
        metrics_dict[f"{mode}_recall_class_{i}"] = recall_per_class[i]
        metrics_dict[f"{mode}_f1_class_{i}"] = f1_per_class[i]

    wandb.log(metrics_dict)

    # Reset metrics
    metric_acc.reset()
    metric_prec.reset()
    metric_rec.reset()
    metric_f1.reset()
    metric_acc_per_class.reset()
    metric_prec_per_class.reset()
    metric_rec_per_class.reset()
    metric_f1_per_class.reset()

    return {
        "loss": loss_avg,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_per_class": accuracy_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class
    }
