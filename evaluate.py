import torch
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm


import torch
import numpy as np
from tqdm import tqdm

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_file: str = "predictions"
):
    """
    Predicts the labels for a given dataset and saves them to an NPZ file.

    Args:
        model: The neural network model.
        dataloader: The DataLoader for the dataset.
        device: The device to run prediction on (CPU/GPU).
        output_file: The file to save the predictions.

    Returns:
        A NumPy array of logits.
    """
    model.eval()
    sample_ids = []
    logits_list = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Prediction', unit='batch')
        for inputs, sample_id in progress_bar:  # Assuming dataloader returns sample_id
            inputs = inputs.to(device)
            outputs = model(inputs)  # Logits before sigmoid

            sample_ids.extend(sample_id.cpu().numpy())  # Convert sample IDs to NumPy
            logits_list.append(outputs.cpu().numpy())  # Convert logits to NumPy

    logits = np.concatenate(logits_list, axis=0)

    # Save to NPZ file
    np.savez_compressed(f"{output_file}", sample_id=np.array(sample_ids), logits=logits)
    

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
    metric_acc = Accuracy(task="multilabel", num_labels=num_classes, average="macro").to(device)
    metric_prec = Precision(task="multilabel", num_labels=num_classes, average="macro").to(device)
    metric_rec = Recall(task="multilabel", num_labels=num_classes, average="macro").to(device)
    metric_f1 = F1Score(task="multilabel", num_labels=num_classes, average="macro").to(device)

    metric_acc_per_class = Accuracy(task="multilabel", num_labels=num_classes, average="none").to(device)
    metric_prec_per_class = Precision(task="multilabel", num_labels=num_classes, average=None).to(device)
    metric_rec_per_class = Recall(task="multilabel", num_labels=num_classes, average=None).to(device)
    metric_f1_per_class = F1Score(task="multilabel", num_labels=num_classes, average=None).to(device)

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
        f"{mode}/epoch": epoch + 1,
        f"{mode}/{mode}_loss": loss_avg,
        f"{mode}/{mode}_accuracy": accuracy,
        f"{mode}/{mode}_precision": precision,
        f"{mode}/{mode}_recall": recall,
        f"{mode}/{mode}_f1": f1
    }

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
