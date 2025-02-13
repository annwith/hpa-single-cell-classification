import argparse
import typing as tp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import wandb

from utils import save_checkpoint, load_checkpoint, \
    train_valid_split_multilabel, train_transformations, valid_transformations
from dataset import HPADatasetFourChannelsImages
from models import HPAClassifier


def train_model(
    dataset_channels: int,
    dataset_path: str,
    labels_path: str,
    class_weights: tp.Optional[tp.List[float]],
    architecture: str,
    pretrained_weights_path: tp.Optional[str],
    batch_size: int,
    epochs: int,
    accumulate_steps: int,
    learning_rate: float,
    save_checkpoint_path: str,
    resume_checkpoint_path: tp.Optional[str] = None,
    project_name: str = "hpa-project",  # wandb project name
    run_name: str = "experiment-1"  # wandb run name
):
    """
    Train a model using the given parameters.

    Parameters:
    - dataset_channels: Number of dataset channels
    - dataset_path: Path to the dataset directory
    - labels_path: Path to the CSV file with labels
    - class_weights: Class weights
    - architecture: Model architecture
    - pretrained_weights_path: Path to the pre-trained weights
    - batch_size: Batch size
    - epochs: Number of epochs to train the model
    - accumulate_steps: Number of batches to accumulate gradients before updating weights
    - learning_rate: Learning rate
    - save_checkpoint_path: Path to save the checkpoint
    - resume_checkpoint_path: Path to the checkpoint to resume training
    - project_name: wandb project name
    - run_name: wandb run name
    """

    if dataset_channels != 4:
        print("Only 4 channels dataset is supported")
        raise ValueError("Only 4 channels dataset is supported")
    
    print("\nDataset information:")
    # Load the dataset
    train, valid = train_valid_split_multilabel(
        hpa_dataset_class=HPADatasetFourChannelsImages,
        dataset_dir=dataset_path,
        labels_csv=labels_path,
        train_transform=train_transformations(),
        valid_transform=valid_transformations(),
        test_size=0.10,
    )

    # Print the number of samples in the train and valid datasets
    print(f"\nTrain dataset: {len(train)} samples")
    print(f"Valid dataset: {len(valid)} samples")
    print(f"Shape: {train[0][0].shape}")

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    # Load the model
    model = HPAClassifier(
        backbone=architecture,
        pretrained_weights_path=pretrained_weights_path,
        num_classes=19,
        in_channels=4)

    # Define the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}") # Print the device

    # Convert the class weights to a tensor
    print("\nClass weights:", class_weights)
    if class_weights:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Put the model on the device
    model.to(device)

    # Initialize wandb
    wandb.init(
        project=project_name, 
        name=run_name, 
        mode="offline")
    
    wandb.config.update({
        "dataset_channels": dataset_channels,
        "dataset_path": dataset_path,
        "labels_path": labels_path,
        "class_weights": class_weights,
        "epochs": epochs,
        "batch_size": batch_size,
        "accumulate_steps": accumulate_steps,
        "learning_rate": optimizer.defaults["lr"],
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "architecture": architecture
    })

    # Initialize loss and epoch variables
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_checkpoint_path:
        start_epoch, _ = load_checkpoint(
            model=model,
            optimizer=optimizer,
            filename=resume_checkpoint_path)

    # Training loop for each epoch
    for epoch in range(start_epoch, epochs):  # Loop through all epochs

        model.train()  # Set model to training mode (affects layers like dropout, batchnorm)
        running_loss = 0.0  # Initialize running loss to accumulate loss over batches
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')  # Progress bar for tracking batches during training
        optimizer.zero_grad()  # Initialize gradients to zero before starting the loop (done outside loop to avoid redundancy)

        # Loop through batches of the training set
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the specified device (GPU/CPU)

            outputs = model(inputs) # Forward pass: Get predictions from the model
            loss = criterion(outputs, labels) # Calculate loss between predictions and true labels
            loss = loss / accumulate_steps # Normalize the loss by dividing it by accumulate_steps to ensure gradient accumulation is balanced
            loss.backward() # Backpropagation: Compute gradients for the loss with respect to the model's weights

            running_loss += loss.item()  # Accumulate the loss value for this batch

            # Update weights every `accumulate_steps` batches or at the last batch
            if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()  # Update the weights based on accumulated gradients
                optimizer.zero_grad()  # Reset gradients after updating weights to prepare for the next accumulation

                # Log metrics to W&B after weights are updated
                wandb.log({
                    "epoch": epoch + 1,  # Log the current epoch number
                    "train_loss": running_loss  # Log the accumulated training loss
                })

                running_loss = 0.0  # Reset the running loss after logging

            # Update progress bar with current average training loss for the batch
            progress_bar.set_postfix({'Train Loss': running_loss / len(progress_bar)})

        # Validation loop (no gradients needed here, so we disable gradient computation)
        model.eval()  # Set model to evaluation mode (affects dropout, batchnorm behavior)
        running_loss = 0.0  # Initialize running loss for validation

        with torch.no_grad():  # Disable gradient computation for validation (faster and saves memory)
            progress_bar = tqdm(valid_loader, desc='Validation', unit='batch')  # Progress bar for tracking validation batches
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device

                outputs = model(inputs)  # Forward pass for validation data
                loss = criterion(outputs, labels)  # Calculate the loss for validation
                running_loss += loss.item()  # Accumulate validation loss

                # Update progress bar with the current validation loss
                progress_bar.set_postfix({'Valid Loss': running_loss / len(progress_bar)})

            # Calculate the average validation loss for this epoch
            valid_loss = running_loss / len(valid_loader)

        # Log validation loss to W&B at the end of each epoch
        wandb.log({
            "epoch": epoch + 1,  # Log the current epoch number
            "valid_loss": valid_loss  # Log the average validation loss for this epoch
        })

        # Save checkpoint after every epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=valid_loss,
            filename=save_checkpoint_path)
        wandb.save(f'checkpoint_epoch_{epoch}.pth')  # Save model to W&B

        # Print the training and validation loss for this epoch
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}")

    # Log the final model after training
    wandb.save('final_model.pth')

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    # Parser for command-line arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset_channels', type=int, default=4, help='Number of dataset channels')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset directory')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the CSV file with labels')
    parser.add_argument('--class_weights', type=str, default=None, help="Comma-separated list of class weights")
    parser.add_argument('--architecture', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--pretrained_weights_path', type=str, default=None, help='Path to the pre-trained weights')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--accumulate_steps', type=int, default=1, help='Number of batches to accumulate gradients before updating weights')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_checkpoint_path', type=str, default='checkpoint.pth', help='Path to save the checkpoint')
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='Path to the checkpoint to resume training')

    args = parser.parse_args()

    # Print CUDA and cuDNN versions
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")  # Print the CUDA version
    print(f"cuDNN version: {cudnn.version()}")  # Print the cuDNN version

    # Print formatted arguments
    print("\nTraining arguments:")

    print(f"{'Dataset Channels:':<25} {args.dataset_channels}")
    print(f"{'Dataset Path:':<25} {args.dataset_path}")
    print(f"{'Labels Path:':<25} {args.labels_path}")
    print(f"{'Class Weights:':<25} {args.class_weights}")
    
    print(f"{'Architecture:':<25} {args.architecture}")
    print(f"{'Pretrained Weights Path:':<25} {args.pretrained_weights_path if args.pretrained_weights_path else 'None'}")

    print(f"{'Epochs:':<25} {args.epochs}")
    print(f"{'Batch Size:':<25} {args.batch_size}")
    print(f"{'Accumulate Steps:':<25} {args.accumulate_steps}")
    print(f"{'Learning Rate:':<25} {args.learning_rate}")

    print(f"{'Save Checkpoint Path:':<25} {args.save_checkpoint_path}")
    print(f"{'Resume Checkpoint Path:':<25} {args.resume_checkpoint_path if args.resume_checkpoint_path else 'None'}")

    # Check if any argument is the string 'none'
    print("\nSetting the following arguments to None:")
    for arg in vars(args):
        if str(vars(args)[arg]).lower() == 'none':  # Check if it is the string 'none'
            vars(args)[arg] = None
            print(f"Setting {arg} to None")

    if args.class_weights:
        class_weights_list = list(map(float, args.class_weights.split(",")))
    else:
        class_weights_list = None

    # Train the model
    train_model(
        dataset_channels=args.dataset_channels,
        dataset_path=args.dataset_path,
        labels_path=args.labels_path,
        class_weights=class_weights_list,
        architecture=args.architecture,
        pretrained_weights_path=args.pretrained_weights_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate_steps=args.accumulate_steps,
        learning_rate=args.learning_rate,
        save_checkpoint_path=args.save_checkpoint_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        project_name="hpa-project",
        run_name="experiment-1"
    )
