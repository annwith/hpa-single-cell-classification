import argparse
import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import wandb

from utils import save_checkpoint, load_checkpoint, \
    train_valid_split_multilabel, train_transformations, valid_transformations, \
    print_metrics
from dataset import HPADataset, HPADatasetFourChannelsImages
from models import HPAClassifier
from train import train_epoch
from evaluate import evaluate


def select_optimizer(
    optimizer_name: str, 
    model: nn.Module, 
    learning_rate: float
) -> tp.Union[optim.Adam, optim.AdamW, optim.SGD]:
    """
    Select the optimizer based on the given name.

    Parameters:
    - optimizer_name: Name of the optimizer
    - model: Model
    - learning_rate: Learning rate

    Returns:
    - Optimizer
    """
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9)
    else:
        raise ValueError("Invalid optimizer")
    
    return optimizer


def train_model(
    dataset_channels: int,
    dataset_path: str,
    labels_path: str,
    publichpa_labels_path: str,
    image_normalization: str,
    class_weights: tp.Optional[tp.List[float]],
    architecture: str,
    pretrained_weights_path: tp.Optional[str],
    batch_size: int,
    epochs: int,
    accumulate_steps: int,
    learning_rate: float,
    optimizer_name: str,
    scheduler_eta_min: float = 1e-6,
    save_checkpoint_path: str,
    resume_checkpoint_path: tp.Optional[str] = None,
    wandb_project_name: str = 'hpa-project',
    wandb_entity_name: str = 'hpa-team',
    wandb_run_name: str = 'experiment',
    wandb_mode: str = 'offline'
):
    """
    Train a model using the given parameters.

    Parameters:
    - dataset_channels: Number of dataset channels
    - dataset_path: Path to the dataset directory
    - labels_path: Path to the CSV file with labels
    - publichpa_labels_path: Path to the CSV file with labels
    - class_weights: Class weights
    - architecture: Model architecture
    - pretrained_weights_path: Path to the pre-trained weights
    - batch_size: Batch size
    - epochs: Number of epochs to train the model
    - accumulate_steps: Number of batches to accumulate gradients before updating weights
    - learning_rate: Learning rate
    - optimizer_name: Optimizer
    - save_checkpoint_path: Path to save the checkpoint
    - resume_checkpoint_path: Path to the checkpoint to resume training
    - wandb_project_name: wandb project name
    - wandb_entity_name: wandb entity name
    - wandb_run_name: wandb run name
    - wandb_mode: wandb mode
    """

    if dataset_channels == 1:
        # Load the dataset
        train, valid = train_valid_split_multilabel(
            hpa_dataset_class=HPADataset,
            dataset_dir=dataset_path,
            labels_csv=labels_path,
            publichpa_labels_csv=publichpa_labels_path,
            train_transform=train_transformations(image_normalization),
            valid_transform=valid_transformations(image_normalization),
            test_size=0.10,
        )
    elif dataset_channels == 4:
        # Load the dataset
        train, valid = train_valid_split_multilabel(
            hpa_dataset_class=HPADatasetFourChannelsImages,
            dataset_dir=dataset_path,
            labels_csv=labels_path,
            train_transform=train_transformations(image_normalization),
            valid_transform=valid_transformations(image_normalization),
            test_size=0.10,
        )
    else:
        raise ValueError("Invalid number of dataset channels")

    # Print the number of samples in the train and valid datasets
    print("\nDataset size information:")
    print(f"Train dataset: {len(train)} samples")
    print(f"Valid dataset: {len(valid)} samples")

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size, 
        num_workers=4,
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid, 
        batch_size=batch_size, 
        num_workers=4,
        shuffle=False)

    # Load the model
    model = HPAClassifier(
        backbone=architecture,
        pretrained_weights_path=pretrained_weights_path,
        num_classes=19,
        in_channels=4)

    # Define the optimizer
    optimizer = select_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        learning_rate=learning_rate)

    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader),
        eta_min=scheduler_eta_min)  # Smooth decay

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}") # Print the device

    # Define the loss function
    if class_weights:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor.to(device))
        print("\nUsing class weights")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("\nNot using class weights")

    # Put the model on the device
    model.to(device)

    # Initialize wandb
    wandb.init(
        project=wandb_project_name,
        entity=wandb_entity_name,
        name=wandb_run_name,
        mode=wandb_mode)
    
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
        start_epoch = load_checkpoint(
            model=model,
            optimizer=optimizer,
            filename=resume_checkpoint_path,
            device=device,
            scheduler=scheduler)

    # Training loop for each epoch
    for epoch in range(start_epoch, epochs):  # Loop through all epochs
        # Train the model for one epoch
        train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            epochs=epochs,
            accumulate_steps=accumulate_steps,
            wandb=wandb)
        
        # Evaluate the model on the training set
        train_metrics = evaluate(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            mode="train",
            wandb=wandb)
        print_metrics(train_metrics, mode="train") 

        # Evaluate the model on the validation set
        valid_metrics = evaluate(
            model=model,
            criterion=criterion,
            dataloader=valid_loader,
            device=device,
            epoch=epoch,
            mode="valid",
            wandb=wandb)
        print_metrics(valid_metrics, mode="valid")

        # Save the model
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            filename=f'{save_checkpoint_path}/{wandb_run_name}.pth',
            scheduler=scheduler)
        
        # Save the model to W&B
        wandb.save(f'{save_checkpoint_path}/{wandb_run_name}.pth')  # Save model to W&B

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    # Parser for command-line arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset_channels', type=int, default=4, help='Number of dataset channels')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset directory')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the CSV file with labels')
    parser.add_argument('--publichpa_labels_path', type=str, required=True, help='Path to the CSV file with labels')
    parser.add_argument('--image_normalization', type=str, default='imagenet', help='Image normalizer')
    parser.add_argument('--class_weights', type=str, default=None, help="Comma-separated list of class weights")
    parser.add_argument('--architecture', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--pretrained_weights_path', type=str, default=None, help='Path to the pre-trained weights')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--accumulate_steps', type=int, default=1, help='Number of batches to accumulate gradients before updating weights')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer_name', type=str, default='adam', help='Optimizer')
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-6, help='Minimum learning rate for the scheduler')
    parser.add_argument('--save_checkpoint_path', type=str, default='checkpoint.pth', help='Path to save the checkpoint')
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='Path to the checkpoint to resume training')
    parser.add_argument('--wandb_project_name', type=str, default='hpa-project', help='wandb project name')
    parser.add_argument('--wandb_entity_name', type=str, default='hpa-team', help='wandb entity name')
    parser.add_argument('--wandb_run_name', type=str, default='experiment', help='wandb run name')
    parser.add_argument('--wandb_mode', type=str, default='offline', help='wandb mode')

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
    print(f"{'PublicHPA Labels Path:':<25} {args.publichpa_labels_path}")
    print(f"{'Image Normalization:':<25} {args.image_normalization}")
    print(f"{'Class Weights:':<25} {args.class_weights}")
    
    print(f"{'Architecture:':<25} {args.architecture}")
    print(f"{'Pretrained Weights Path:':<25} {args.pretrained_weights_path if args.pretrained_weights_path else 'None'}")

    print(f"{'Epochs:':<25} {args.epochs}")
    print(f"{'Batch Size:':<25} {args.batch_size}")
    print(f"{'Accumulate Steps:':<25} {args.accumulate_steps}")
    print(f"{'Learning Rate:':<25} {args.learning_rate}")
    print(f"{'Optimizer Name:':<25} {args.optimizer_name}")
    print(f"{'Scheduler Eta Min:':<25} {args.scheduler_eta_min}")

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
        print("\nConverting class weights to list.")
    else:
        class_weights_list = None

    # Train the model
    train_model(
        dataset_channels=args.dataset_channels,
        dataset_path=args.dataset_path,
        labels_path=args.labels_path,
        publichpa_labels_path=args.publichpa_labels_path,
        image_normalization=args.image_normalization,
        class_weights=class_weights_list,
        architecture=args.architecture,
        pretrained_weights_path=args.pretrained_weights_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate_steps=args.accumulate_steps,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer_name,
        scheduler_eta_min=args.scheduler_eta_min,
        save_checkpoint_path=args.save_checkpoint_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        wandb_project_name=args.wandb_project_name,
        wandb_entity_name=args.wandb_entity_name,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode
    )
