import argparse
from tqdm import tqdm  # Importar tqdm para a barra de progresso

import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim

from utils import save_checkpoint, load_checkpoint, train_valid_split_multilabel, train_transformations, valid_transformations
from dataset import HPADatasetFourChannelsImages

def train_model(
    dataset_dir: str,
    labels_csv: str,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    model: nn.Module,
    batch_size: int,
    num_epochs: int,
    checkpoint: str,
    resume_checkpoint: tp.Optional[str] = None
):
    '''
    Função para treinar um modelo de classificação de imagens.
    Parâmetros:
        dataset_dir: str - Diretório do dataset
        labels_csv: str - Caminho para o arquivo CSV com os rótulos
        optimizer: optim.Optimizer - Otimizador para treinar o modelo
        criterion: nn.Module - Função de loss para otimizar
        model: nn.Module - Modelo a ser treinado
        batch_size: int - Tamanho do batch
        num_epochs: int - Número de épocas para treinar o modelo
        checkpoint: str - Caminho para salvar o checkpoint
        resume_checkpoint: str - Caminho para o checkpoint para retomar o treinamento
    '''

    # Carregar os dados
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

    # Definir dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializar variáveis de loss e epoch
    train_losses = []
    valid_losses = []
    start_epoch = 0

    # Se for fornecido um checkpoint, carregar os pesos e o otimizador
    if resume_checkpoint:
        start_epoch, train_losses, valid_losses = load_checkpoint(model, optimizer, resume_checkpoint)

    # Loop de treinamento
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        # Adicionar a barra de progresso para os batches no treinamento
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zerar os gradientes
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Propagar o erro
            optimizer.step()  # Atualizar os pesos
            running_loss += loss.item()

            # Atualizar a barra de progresso com a loss do batch atual
            progress_bar.set_postfix({'Train Loss': running_loss / len(progress_bar)})

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Salvar checkpoint após cada época
        save_checkpoint(epoch, model, optimizer, train_losses, valid_losses, checkpoint)

        # Loop de validação
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(valid_loader, desc='Validation', unit='batch')
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Atualizar a barra de progresso com a loss da validação
                progress_bar.set_postfix({'Valid Loss': running_loss / len(progress_bar)})

            valid_loss = running_loss / len(valid_loader)
            valid_losses.append(valid_loss)

        torch.cuda.empty_cache()  # Limpar memória de GPU

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}")


def train_squeeze_cam(
    dataset_dir: str,
    labels_csv: str,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    model: nn.Module,
    batch_size: int,
    weights_update: int,
    num_epochs: int,
    checkpoint: str,
    resume_checkpoint: tp.Optional[str] = None
):
    '''
    Função para treinar um modelo de classificação de imagens.
    Parâmetros:
        dataset_dir: str - Diretório do dataset
        labels_csv: str - Caminho para o arquivo CSV com os rótulos
        optimizer: optim.Optimizer - Otimizador para treinar o modelo
        criterion: nn.Module - Função de loss para otimizar
        model: nn.Module - Modelo a ser treinado
        batch_size: int - Tamanho do batch
        weights_update: int - Número de batches para acumular gradientes antes de atualizar os pesos
        num_epochs: int - Número de épocas para treinar o modelo
        checkpoint: str - Caminho para salvar o checkpoint
        resume_checkpoint: str - Caminho para o checkpoint para retomar o treinamento
    '''

    # Definir dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar os dados
    train, valid = train_valid_split_multilabel(
        hpa_dataset_class=HPADatasetFourChannelsImages,
        dataset_dir=dataset_dir,
        labels_csv=labels_csv,
        train_transform=train_transformations(),
        valid_transform=valid_transformations(),
        test_size=0.20,
    )

    # Loaders
    batch_size = 16

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    # Inicializar variáveis de loss e epoch
    train_losses = []
    valid_losses = []
    start_epoch = 0

    # Se for fornecido um checkpoint, carregar os pesos e o otimizador
    if resume_checkpoint:
        start_epoch, train_losses, valid_losses = load_checkpoint(model, optimizer, resume_checkpoint)

    # Loop de treinamento
    for epoch in range(start_epoch, num_epochs):

        # Modo de treinamento
        model.train()

        # Zerar os gradientes antes do início do treino e inicializar a loss
        optimizer.zero_grad()  
        running_loss = 0.0

        # Adicionar a barra de progresso para os batches no treinamento
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        # Loop de treinamento
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            # Mover os dados para a GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            _, _, outputs = model(inputs)    

            # Calcular a loss
            loss = criterion(outputs, labels)

            # Acumular gradientes
            loss.backward()

            # Atualizar a loss total
            running_loss += loss.item()

            # Atualizar a barra de progresso com a loss do batch atual
            progress_bar.set_postfix({'Train Loss': running_loss / (batch_idx + 1)})

            # Atualizar os pesos após 'weights_update' batches
            if (batch_idx + 1) % weights_update == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()  # Atualizar os pesos
                optimizer.zero_grad()  # Zerar os gradientes acumulados

        # Calcular a loss média do treinamento
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Salvar checkpoint após cada época
        save_checkpoint(epoch, model, optimizer, train_losses, valid_losses, checkpoint)

        # Loop de validação
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(valid_loader, desc='Validation', unit='batch')
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                _, _, outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Atualizar a barra de progresso com a loss da validação
                progress_bar.set_postfix({'Valid Loss': running_loss / (batch_idx + 1)})

            valid_loss = running_loss / len(valid_loader)
            valid_losses.append(valid_loss)

        torch.cuda.empty_cache()  # Limpar memória de GPU

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}")


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
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    if args.model == 'efficientnet-b0':
        from models import HPA_EfficientNet_B0_Model
        model_arq = HPA_EfficientNet_B0_Model()
    elif args.model == 'squeezenet-cam':
        from models import SqueezeNetCAM
        model_arq = SqueezeNetCAM()

        # SqueezeNet
        # Congelar todas as camadas por padrão
        for param in model_arq.parameters():
            param.requires_grad = False

        # Descongelar as camadas do classificador
        for param in model_arq.squeezenet.classifier.parameters():
            param.requires_grad = True

        # Definir o otimizador apenas para os parâmetros descongelados
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_arq.parameters()),
            lr=args.lr
        )

        # Definir a função de perda (criterion)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Chama a função de treinamento com os parâmetros recebidos
    train_squeeze_cam(
        dataset_dir=args.dataset_dir,
        labels_csv=args.labels_csv,
        optimizer=optimizer,
        criterion=criterion,
        model=model_arq,
        batch_size=args.batch_size,
        weights_update=args.weights_update,
        num_epochs=args.epochs,
        checkpoint=args.checkpoint,
        resume_checkpoint=args.resume
    )
