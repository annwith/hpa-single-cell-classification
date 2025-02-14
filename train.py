import torch
from tqdm import tqdm

def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: torch.device,
    epoch: int,
    epochs: int,
    accumulate_steps: int,
    wandb):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        train_loader: The DataLoader for training data.
        criterion: The loss function.
        optimizer: The optimizer for updating model weights.
        scheduler: The learning rate scheduler.
        device: The device to run training on (CPU/GPU).
        epoch: The current epoch number.
        epochs: Total number of epochs for training.
        accumulate_steps: Number of steps for gradient accumulation.
        wandb: The Weights & Biases object for logging.

    Returns:
        Average training loss for the epoch.
    """
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients
    running_loss = 0.0  # Initialize loss accumulator
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss = loss / accumulate_steps  # Normalize loss for accumulation
        loss.backward()  # Compute gradients

        running_loss += loss.item()  # Accumulate loss

        # Update weights every `accumulate_steps` batches or at the last batch
        if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Clear gradients for next step

            # Log metrics to W&B
            lr = scheduler.get_last_lr()[0]
            wandb.log({
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": running_loss / accumulate_steps
            })
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss / accumulate_steps}")
            print(f"Learning rate: {lr}")

            # Update progress bar
            progress_bar.set_postfix({'train loss': running_loss / accumulate_steps})

            running_loss = 0.0  # Reset loss accumulator
            scheduler.step()  # Adjust learning rate

    return running_loss / len(train_loader)  # Return average training loss
