import argparse
import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
from models import HPA_EfficientNet_B0_Model
from dataset import train_valid_split_multilabel, train_transformations, valid_transformations


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_losses: tp.List[float],
    valid_losses: tp.List[float],
    filename: str = "checkpoint.pth"
):
    '''
    Salva um checkpoint do modelo, otimizador e outras informações úteis.
    Parâmetros:
        epoch: int - Número da época atual
        model: nn.Module - Modelo a ser salvo
        optimizer: optim.Optimizer - Otimizador a ser salvo
        train_losses: List[float] - Lista de loss no treinamento
        valid_losses: List[float] - Lista de loss na validação
        filename: str - Nome do arquivo para salvar o checkpoint
    '''
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }
    torch.save(checkpoint, filename)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    filename: str = "checkpoint.pth"
):
    '''
    Carrega um checkpoint do modelo e otimizador.
    Parâmetros:
        model: nn.Module - Modelo a ser carregado
        optimizer: optim.Optimizer - Otimizador a ser carregado
        filename: str - Nome do arquivo para carregar o checkpoint
    Retorna:
        epoch: int - Número da última época treinada
        train_losses: List[float] - Lista de loss no treinamento
        valid_losses: List[float] - Lista de loss na validação
    '''
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']
    return epoch, train_losses, valid_losses


def train_model(
    dataset_dir: str,
    labels_csv: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint: str,
    resume_checkpoint: tp.Optional[str] = None
):
    ''' 
    Função para treinar um modelo de classificação de imagens.
    Parâmetros:
        dataset_dir: str - Diretório do dataset
        labels_csv: str - Caminho para o arquivo CSV com os rótulos
        num_epochs: int - Número de épocas para treinar o modelo
        batch_size: int - Tamanho do batch
        learning_rate: float - Taxa de aprendizado
        checkpoint: str - Caminho para salvar o checkpoint
        resume_checkpoint: str - Caminho para o checkpoint para retomar o treinamento
    '''

    # Definir dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instanciar o modelo
    model = HPA_EfficientNet_B0_Model()
    model = model.to(device)

    # Definir o otimizador
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Definir a função de perda (criterion)
    criterion = nn.BCEWithLogitsLoss()

    # Carregar os dados
    train, valid = train_valid_split_multilabel(
        dataset_dir=dataset_dir,
        labels_csv=labels_csv,
        train_transform=train_transformations(),
        valid_transform=valid_transformations(),
        test_size=0.25,
    )

    # Loaders
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
    model.train()
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zerar os gradientes
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Propagar o erro
            optimizer.step()  # Atualizar os pesos
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Salvar checkpoint após cada época
        save_checkpoint(epoch, model, optimizer, train_losses, valid_losses, checkpoint)

        # Loop de validação
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            valid_loss = running_loss / len(valid_loader)
            valid_losses.append(valid_loss)

        model.train()  # Retornar ao modo de treinamento


if __name__ == "__main__":
    # Parser para os argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Treinamento de Modelo')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas para treinar o modelo')
    parser.add_argument('--batch', type=int, default=64, help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Taxa de aprendizado')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Diretório do dataset')
    parser.add_argument('--labels_csv', type=str, required=True, help='Caminho para o arquivo CSV com os rótulos')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Caminho para salvar o checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Caminho para o checkpoint para retomar o treinamento')

    args = parser.parse_args()

    # Chama a função de treinamento com os parâmetros recebidos
    train_model(
        dataset_dir=args.dataset_dir,
        labels_csv=args.labels_csv,
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        checkpoint=args.checkpoint,
        resume_checkpoint=args.resume
    )
