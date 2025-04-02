import psutil
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
        outputs = model(inputs)  # Forward pass [Those are raw logits - I think so...]

        loss = criterion(outputs, labels)  # Compute loss
        loss = loss / accumulate_steps  # Normalize loss for accumulation
        loss.backward()  # Compute gradients

        running_loss += loss.item()  # Accumulate loss

        # Update weights every `accumulate_steps` batches or at the last batch
        if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Clear gradients for next step

            # Get current learning rate
            # Note: This assumes a single optimizer with a single parameter group
            lr = optimizer.param_groups[0]["lr"]
            
            # Log training metrics
            wandb.log({
                "train/epoch": epoch + 1,
                "train/lr": lr,
                "train/train_loss": running_loss / accumulate_steps
            })

            # Get CPU and RAM usage
            cpu_mem = psutil.virtual_memory()
            cpu_mem_used = cpu_mem.used / (1024 ** 3)  # Convert to GB
            cpu_mem_free = cpu_mem.available / (1024 ** 3)  # Convert to GB

            # Log GPU metrics
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB

            wandb.log({
                "System/GPU Memory Allocated (GB)": gpu_memory_allocated,
                "System/GPU Memory Reserved (GB)": gpu_memory_reserved,
                "System/CPU Memory Used (GB)": cpu_mem_used,
                "System/CPU Memory Free (GB)": cpu_mem_free
            })

            # Update progress bar
            progress_bar.set_postfix({
                'train loss': running_loss / accumulate_steps,
                'lr': lr
            })

            running_loss = 0.0  # Reset loss accumulator
            scheduler.step()  # Adjust learning rate