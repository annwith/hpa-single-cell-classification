import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from models import HPA_EfficientNet_B0_Model
from dataset import train_valid_split_multilabel, train_transformations, \
    valid_transformations

def train_model(
    dataset_dir: str,
    labels_csv: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float
):
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

    # Loop de treinamento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            print("Batch")

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("Treinamento completo!")

    # Loop de validação
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

        print(f"Validação, Loss: {running_loss/len(valid_loader):.4f}")

    print("Validação completa!")

if __name__ == "__main__":
    # Diretório do dataset e arquivo CSV com os rótulos
    dataset_dir = '/mnt/ssd/hpa-single-cell-image-classification/train'
    labels_csv = '/mnt/ssd/hpa-single-cell-image-classification/train.csv'

    # Parser para os argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Treinamento de Modelo')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas para treinar o modelo')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Taxa de aprendizado')

    args = parser.parse_args()

    # Chama a função de treinamento com os parâmetros recebidos
    train_model(
        dataset_dir=dataset_dir,
        labels_csv=labels_csv,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
