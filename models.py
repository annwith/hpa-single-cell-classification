import torch
import torch.nn as nn
from torchvision import models
import timm  # For loading ResNet, EfficientNet easily


class HPAClassifier(nn.Module):
    """
    Classifier for the Human Protein Atlas dataset using a pretrained model from timm.
    """

    def __init__(self, backbone="resnet50", num_classes=19, pretrained_weights_path=None, in_channels=4):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)

        # Load weights from a file
        if pretrained_weights_path is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(pretrained_weights_path, map_location=device)
            self.model.load_state_dict(state_dict)

            print("\nPretrained weights loaded successfully!")

        # Adjust first convolutional layer to take 4-channel input
        first_layer = self.model.conv1  # Works for ResNet
        new_conv = nn.Conv2d(in_channels, first_layer.out_channels, 
                                 kernel_size=first_layer.kernel_size, 
                                 stride=first_layer.stride, 
                                 padding=first_layer.padding, 
                                 bias=first_layer.bias is not None)
        
        # Copia os pesos da camada antiga para os 3 primeiros canais da nova camada
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = first_layer.weight  # Copia pesos originais (RGB)
            new_conv.weight[:, 3:4, :, :] = first_layer.weight[:, 0:1, :, :]  # Copia o canal R para o 4º canal

        self.model.conv1 = new_conv  # Substitui a camada antiga pela nova

    def forward(self, x):
        return self.model(x)


def HPA_EfficientNet_B0(num_classes: int, weights: str = None, weights_path: str = None):
    '''
    Builds a model for the Human Protein Atlas dataset using EfficientNet-B0.

    Parameters:
        num_classes: int
            The number of classes in the dataset.
        weights: str
            The weights to use for the EfficientNet-B0 model.
        weights_path: str
            The path to the weights file to load.
    Returns:
        nn.Module
    '''

    # Carregar o modelo EfficientNet-B0 pré-treinado
    model = models.efficientnet_b0(weights=weights)

    # Obter a primeira camada convolucional
    first_conv_layer = model.features[0][0]

    # Criar uma nova camada convolucional com 4 canais de entrada, mantendo os outros parâmetros
    new_conv_layer = nn.Conv2d(
        in_channels=4,  # Mudar para 4 canais de entrada
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=first_conv_layer.bias is not None
    )

    # Copiar os pesos dos 3 primeiros canais da camada original para a nova camada
    with torch.no_grad():
        new_conv_layer.weight[:, :3, :, :] = first_conv_layer.weight

        # Duplicate the first channel's weights into the fourth channel
        new_conv_layer.weight[:, 3, :, :] = first_conv_layer.weight[:, 0, :, :]

    # Substituir a camada original pela nova
    model.features[0][0] = new_conv_layer

    # Modificar a última camada para classificação multirrótulo
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # Aplicar a função sigmoid na saída para multirrótulo
    model.add_module('sigmoid', torch.nn.Sigmoid())

    if weights_path is not None:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model


class SqueezeNetCAM(nn.Module):
    def __init__(self, num_classes=19):
        super(SqueezeNetCAM, self).__init__()

        # Carregar o modelo SqueezeNet
        self.squeezenet = models.squeezenet1_1(weights=None)

        # Obter os pesos da camada original com 3 canais
        original_conv1_weights = self.squeezenet.features[0].weight.data

        # Modificar a primeira camada convolucional para aceitar 4 canais em vez de 3
        self.squeezenet.features[0] = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)

        # Inicializar a nova camada com os pesos da camada original
        with torch.no_grad():
            # Copiar os pesos dos 3 canais pré-treinados e repetir/adaptar para o 4º canal
            self.squeezenet.features[0].weight[:, :3, :, :] = original_conv1_weights
            self.squeezenet.features[0].weight[:, 3, :, :] = torch.mean(original_conv1_weights, dim=1)

        # Remover a camada de classificação original e substituir por uma nova para 19 classes
        self.squeezenet.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

        # Camada de pool adaptativa para combinar com o CAM
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Carregar o state_dict salvo
        state_dict = torch.load('weights/squeezenet_cam.pth', weights_only=True)

        # Remover o prefixo 'squeezenet.' das chaves
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('squeezenet.', '')  # Remover o prefixo
            new_state_dict[new_key] = state_dict[key]

        # Agora carregar o novo state_dict no modelo
        self.squeezenet.load_state_dict(new_state_dict)

    def forward(self, x):
        # Extrair features da última camada convolucional
        features = self.squeezenet.features(x)

        # Aplicar a camada de classificação para obter os mapas de ativação
        output = self.squeezenet.classifier(features)

        # Aplicar global average pooling para transformar o output em [batch_size, num_classes]
        pooled_output = self.global_avg_pool(output)
        pooled_output = pooled_output.view(pooled_output.size(0), -1)

        return output, features, pooled_output
