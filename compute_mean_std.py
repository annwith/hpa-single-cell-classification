import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.v2 as transforms

from utils import train_valid_split_multilabel, compute_mean_std
from dataset import HPADataset, HPADatasetFourChannelsImages


def train_transformations() -> transforms.Compose:
    '''
    Returns a composition of transformations to be applied to the training images.
    Returns:
        transforms.Compose
            The composition of transformations.
    '''
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True)
    ])


def valid_transformations() -> transforms.Compose:
    '''
    Returns a composition of transformations to be applied to the validation images.
    Returns:
        transforms.Compose
            The composition of transformations.
    '''
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True)
    ])


def compute(
    dataset_channels: int,
    dataset_path: str,
    labels_path: str,
    publichpa_labels_path: str,
    batch_size: int,
):
    """
    Train a model using the given parameters.

    Parameters:
    - dataset_channels: Number of dataset channels
    - dataset_path: Path to the dataset directory
    - labels_path: Path to the CSV file with labels
    - publichpa_labels_path: Path to the CSV file with labels
    - batch_size: Batch size
    """

    if dataset_channels == 1:
        # Load the dataset
        train, valid = train_valid_split_multilabel(
            hpa_dataset_class=HPADataset,
            dataset_dir=dataset_path,
            labels_csv=labels_path,
            publichpa_labels_csv=publichpa_labels_path,
            train_transform=train_transformations(),
            valid_transform=valid_transformations(),
            test_size=0.10,
        )
    elif dataset_channels == 4:
        # Load the dataset
        train, valid = train_valid_split_multilabel(
            hpa_dataset_class=HPADatasetFourChannelsImages,
            dataset_dir=dataset_path,
            labels_csv=labels_path,
            train_transform=train_transformations(),
            valid_transform=valid_transformations(),
            test_size=0.10,
        )
    else:
        raise ValueError("Invalid number of dataset channels")

    # Print the number of samples in the train and valid datasets
    print("\nDataset size information:")
    print(f"Train dataset: {len(train)} samples")
    print(f"Valid dataset: {len(valid)} samples")

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    # Compute mean and standard deviation
    mean, std = compute_mean_std(train_loader)

    print(f"\nMean: {mean}")
    print(f"Std: {std}")
    

if __name__ == "__main__":
    # Parser for command-line arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset_channels', type=int, default=4, help='Number of dataset channels')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset directory')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the CSV file with labels')
    parser.add_argument('--publichpa_labels_path', type=str, required=True, help='Path to the CSV file with labels')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

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

    print(f"{'Batch Size:':<25} {args.batch_size}")

    # Train the model
    compute(
        dataset_channels=args.dataset_channels,
        dataset_path=args.dataset_path,
        labels_path=args.labels_path,
        publichpa_labels_path=args.publichpa_labels_path,
        batch_size=args.batch_size
    )
