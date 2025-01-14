import argparse
from tqdm import tqdm  # Importar tqdm para a barra de progresso

import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim

from utils import save_checkpoint, load_checkpoint, train_valid_split_multilabel, train_transformations, valid_transformations
from dataset import HPADatasetFourChannelsImages
from models import HPA_EfficientNet_B0_Model, SqueezeNetCAM

import wandb


def train_model(
    dataset_dir: str,
    labels_csv: str,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    model: nn.Module,
    batch_size: int,
    num_epochs: int,
    checkpoint: str,
    resume_checkpoint: tp.Optional[str] = None,
    project_name: str = "image_classification_project",  # wandb project name
    run_name: str = "experiment_1"  # wandb run name
):
    '''
    Function to train an image classification model with wandb integration.
    '''

    # Initialize wandb
    wandb.init(project=project_name, name=run_name)
    wandb.config.update({
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": optimizer.defaults["lr"]
    })

    # Load the dataset
    train, valid = train_valid_split_multilabel(
        hpa_dataset_class=HPADatasetFourChannelsImages,
        dataset_dir=dataset_dir,
        labels_csv=labels_csv,
        train_transform=train_transformations(),
        valid_transform=valid_transformations(),
        test_size=0.20,
    )

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize loss and epoch variables
    train_losses = []
    valid_losses = []
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_checkpoint:
        start_epoch, train_losses, valid_losses = load_checkpoint(model, optimizer, resume_checkpoint)

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
    parser = argparse.ArgumentParser(description='Treinamento de Modelo')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas para treinar o modelo')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamanho do batch')
    parser.add_argument('--weights_update', type=int, default=1, help='Número de batches para acumular gradientes antes de atualizar os pesos')
    parser.add_argument('--lr', type=float, default=0.001, help='Taxa de aprendizado')
    parser.add_argument('--model', type=str, default='efficientnet-b0', help='Modelo a ser treinado')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Diretório do dataset')
    parser.add_argument('--labels_csv', type=str, required=True, help='Caminho para o arquivo CSV com os rótulos')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Caminho para salvar o checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Caminho para o checkpoint para retomar o treinamento')

    args = parser.parse_args()

    # Pesos das classes
    class_weights = [0.1, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 10.0, 1.0, 0.5, 0.5, 5.0, 0.2, 0.5, 1.0]

    # Converter para torch tensor e garantir que seja do tipo float
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    if args.model == 'efficientnet-b0':
        model_arq = HPA_EfficientNet_B0_Model()

    elif args.model == 'squeezenet-cam':
        model_arq = SqueezeNetCAM(num_classes=19)

        # SqueezeNet
        # Congelar todas as camadas por padrão
        for param in model_arq.parameters():
            param.requires_grad = False

        # Descongelar as camadas do classificador
        for param in model_arq.squeezenet.classifier.parameters():
            param.requires_grad = True

    # Definir o otimizador apenas para os parâmetros descongelados
    model_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_arq.parameters()),
        lr=args.lr
    )

    # Definir a função de perda (criterion)
    model_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Chama a função de treinamento com os parâmetros recebidos
    train_model(
        dataset_dir=args.dataset_dir,
        labels_csv=args.labels_csv,
        optimizer=model_optimizer,
        criterion=model_criterion,
        model=model_arq,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        checkpoint=args.checkpoint,
        resume_checkpoint=args.resume,
        project_name="image_classification_project",
        run_name="experiment_1"
    )
