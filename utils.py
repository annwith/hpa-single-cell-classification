import numpy as np
import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split

from dataset import HPADataset, classes_map


def train_valid_split_multilabel(
    hpa_dataset_class: tp.Type[HPADataset],
    dataset_dir: str,
    labels_csv: str,
    train_transform: tp.Optional[transforms.Compose] = None,
    valid_transform: tp.Optional[transforms.Compose] = None,
    test_size=0.25,
    random_state=42
):
    '''
    Splits the dataset into training and validation sets.
    Parameters:
        hpa_dataset_class: Type[HPADataset]
            The class of the dataset to be used.
        dataset_dir: str
            The directory where the images are stored.
        labels_csv: str
            Path to the CSV file containing labels.
        train_transform: Optional[transforms.Compose]
            If not None, the images in the training set will be transformed.
        valid_transform: Optional[transforms.Compose]
            If not None, the images in the validation set will be transformed.
        test_size: float
            The proportion of the dataset to include in the validation set.
        random_state: int
            The seed used by the random number generator.
    Returns:
        Tuple[HPADataset, HPADataset]
    '''

    # Carregar o dataset completo
    dataset = hpa_dataset_class(images_dir=dataset_dir, labels_csv=labels_csv)
    dataset_size = len(dataset)

    # Obter os índices de todas as amostras
    indices = list(range(dataset_size))

    # Criar uma matriz binária para representar a presença de cada rótulo em cada amostra
    binary_labels = np.zeros((dataset_size, len(classes_map)))

    for i, labels in enumerate(dataset.labels.values.flatten()):
        binary_labels[i, labels] = 1

    # Agora usar esta matriz binária para estratificação
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=binary_labels.sum(axis=1), # Estratificar baseado no número de rótulos por amostra
        random_state=random_state
    )

    train_dataset = hpa_dataset_class(
        images_dir=dataset_dir,
        labels_csv=labels_csv,
        indices=train_indices,
        transform=train_transform)

    valid_dataset = hpa_dataset_class(
        images_dir=dataset_dir,
        labels_csv=labels_csv,
        indices=valid_indices,
        transform=valid_transform)
    
    # Count class occurrences
    def get_label_counts(dataset):
        label_counts = dataset.binary_labels.sum(dim=0).int().tolist()
        return {classes_map[i]: count for i, count in enumerate(label_counts)}

    # Print sorted class distributions
    print("\nSorted label counts: (train)")
    train_label_counts = get_label_counts(train_dataset)
    for label, count in sorted(train_label_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"{label}: {count}")

    print("\nSorted label counts: (valid)")
    valid_label_counts = get_label_counts(valid_dataset)
    for label, count in sorted(valid_label_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"{label}: {count}")

    return train_dataset, valid_dataset


def train_transformations() -> transforms.Compose:
    '''
    Returns a composition of transformations to be applied to the training images.
    Returns:
        transforms.Compose
            The composition of transformations.
    '''
    return transforms.Compose([
        transforms.ToImage(), # Transformar de tensor para imagem
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.0807, 0.0804, 0.0538, 0.0522], std=[0.1233, 0.1273, 0.1422, 0.0818])
    ])


def valid_transformations() -> transforms.Compose:
    '''
    Returns a composition of transformations to be applied to the validation images.
    Returns:
        transforms.Compose
            The composition of transformations.
    '''
    return transforms.Compose([
        transforms.ToImage(), # Transformar de tensor para imagem
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.0807, 0.0804, 0.0538, 0.0522], std=[0.1233, 0.1273, 0.1422, 0.0818])
    ])


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    filename: str,
    scheduler: optim.lr_scheduler.LRScheduler = None
):
    """
    Save the model checkpoint to the specified file.

    Parameters:
        epoch (int): The current epoch.
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer used in training.
        filename (str): Path to save the checkpoint.
        scheduler (optim.lr_scheduler.LRScheduler, optional): The LR scheduler.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }

    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} -> {filename}")

  
def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    filename: str,
    device: torch.device,
    scheduler: optim.lr_scheduler.LRScheduler = None
):
    """
    Load the model, optimizer, and scheduler state from a checkpoint.

    Parameters:
        model (nn.Module): The model to load the state into.
        optimizer (optim.Optimizer): The optimizer to load the state into.
        filename (str): Path to the checkpoint file.
        device (torch.device): Device to map the checkpoint to.
        scheduler (optim.lr_scheduler.LRScheduler, optional): The scheduler.

    Returns:
        int: Last trained epoch.
    """
    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    
    print(f"Checkpoint loaded from epoch {epoch} -> {filename}")

    return epoch


def get_mean_std(loader):
    """
    Calcula a média e o desvio padrão dos canais de um DataLoader.

    Parâmetros:
        loader: DataLoader - DataLoader com as imagens a serem calculadas a média e desvio padrão

    Retorna:
        mean: torch.Tensor - Média dos canais
        std: torch.Tensor - Desvio padrão dos canais
    """

    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data, _ in loader:
        batch_samples = data.size(0)  # Número de imagens no batch
        data = data.view(batch_samples, data.size(1), -1)  # (batch, canais, width * height)
        mean += data.mean(2).sum(0)  # Média por canal
        std += data.std(2).sum(0)    # Desvio padrão por canal
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std