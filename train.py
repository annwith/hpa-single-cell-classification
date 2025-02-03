import argparse
import typing as tp
from tqdm import tqdm  # Importar tqdm para a barra de progresso

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import wandb

from utils import save_checkpoint, load_checkpoint, train_valid_split_multilabel, train_transformations, valid_transformations
from dataset import HPADatasetFourChannelsImages
from models import HPAClassifier


def train_model(
    dataset_path: str,
    labels_path: str,
    learning_rate: float,
    class_weights: torch.Tensor,
    architecture: str,
    batch_size: int,
    num_epochs: int,
    pretrained_weights_path: tp.Optional[str],
    save_checkpoint_path: str,
    resume_checkpoint_path: tp.Optional[str] = None,
    project_name: str = "hpa-project",  # wandb project name
    run_name: str = "experiment-1"  # wandb run name
):
    """
    Train a model using the given parameters.

    Parameters:
        dataset_path: str
            The path to the dataset.
        labels_path: str
            The path to the labels CSV file.
        learning_rate: float
            The learning rate for the optimizer.
        class_weights: torch.Tensor
            The class weights tensor.
        architecture: str
            The architecture to use.
        batch_size: int
            The batch size.
        num_epochs: int
            The number of epochs to train the model.
        save_checkpoint_path: str
            The path to save the checkpoint.
        resume_checkpoint_path: str
            The path to the checkpoint to resume training from.
        project_name: str
            The name of the wandb project.
        run_name: str
            The name of the wandb run.
    """

    # Load the dataset
    train, valid = train_valid_split_multilabel(
        hpa_dataset_class=HPADatasetFourChannelsImages,
        dataset_dir=dataset_path,
        labels_csv=labels_path,
        train_transform=train_transformations(),
        valid_transform=valid_transformations(),
        test_size=0.10,
    )

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    # Load the model
    model = HPAClassifier(
        backbone=architecture,
        pretrained_weights_path=pretrained_weights_path,
        num_classes=19, 
        in_channels=4)

    # Definir o otimizador
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate)

    # Definir a função de perda (criterion)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}") # Print the device

    # Put the model on the device
    model.to(device)

    # Initialize wandb
    wandb.init(project=project_name, name=run_name, mode="offline")
    wandb.config.update({
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": optimizer.defaults["lr"]
    })

    # Initialize loss and epoch variables
    train_losses = []
    valid_losses = []
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_checkpoint_path:
        start_epoch, train_losses, valid_losses = load_checkpoint(model, optimizer, resume_checkpoint_path)

    best_valid_loss = float("inf")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix({'Train Loss': running_loss / len(progress_bar)})

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(valid_loader, desc='Validation', unit='batch')
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                progress_bar.set_postfix({'Valid Loss': running_loss / len(progress_bar)})

            valid_loss = running_loss / len(valid_loader)
            valid_losses.append(valid_loss)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        })

        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_path = f"{checkpoint}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            wandb.save(best_model_path)  # Save the best model to wandb

        # Save checkpoint after every epoch
        checkpoint_path = f"{checkpoint}.pth"
        save_checkpoint(epoch, model, optimizer, train_losses, valid_losses, checkpoint_path)
        wandb.save(checkpoint_path)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    # Parser para os argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Treinamento')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas para treinar o modelo')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamanho do batch')
    parser.add_argument('--weights_update', type=int, default=1, help='Número de batches para acumular gradientes antes de atualizar os pesos')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Taxa de aprendizado')
    parser.add_argument('--architecture', type=str, default='resnet50', help='Arquitetura')
    parser.add_argument('--dataset_path', type=str, required=True, help='Diretório do dataset')
    parser.add_argument('--labels_path', type=str, required=True, help='Caminho para o arquivo CSV com os rótulos')
    parser.add_argument('--pretrained_weights_path', type=str, default=None, help='Caminho para os pesos pré-treinados')
    parser.add_argument('--save_checkpoint_path', type=str, default='checkpoint.pth', help='Caminho para salvar o checkpoint')
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='Caminho para o checkpoint para retomar o treinamento')

    args = parser.parse_args()

    # Print CUDA and cuDNN versions
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")  # Print the CUDA version
    print(f"cuDNN version: {cudnn.version()}")  # Print the cuDNN version

    # Print formatted arguments
    print("\nParâmetros de Treinamento:")
    print(f"{'Epochs:':<25} {args.epochs}")
    print(f"{'Batch Size:':<25} {args.batch_size}")
    print(f"{'Weights Update:':<25} {args.weights_update}")
    print(f"{'Learning Rate:':<25} {args.learning_rate}")
    print(f"{'Architecture:':<25} {args.architecture}")
    print(f"{'Dataset Path:':<25} {args.dataset_path}")
    print(f"{'Labels Path:':<25} {args.labels_path}")
    print(f"{'Pretrained Weights Path:':<25} {args.pretrained_weights_path if args.pretrained_weights_path else 'None'}")
    print(f"{'Save Checkpoint Path:':<25} {args.save_checkpoint_path}")
    print(f"{'Resume Checkpoint Path:':<25} {args.resume_checkpoint_path if args.resume_checkpoint_path else 'None'}")

    # Pesos das classes
    class_weights = [0.1, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 10.0, 1.0, 0.5, 0.5, 5.0, 0.2, 0.5, 1.0]

    # Converter para torch tensor e garantir que seja do tipo float
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Chama a função de treinamento com os parâmetros recebidos
    train_model(
        dataset_path=args.dataset_path,
        labels_path=args.labels_path,
        architecture=args.architecture,
        learning_rate=args.learning_rate,
        class_weights=class_weights_tensor,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        pretrained_weights_path=args.pretrained_weights_path,
        save_checkpoint_path=args.save_checkpoint_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        project_name="hpa-project",
        run_name="experiment-1"
    )
