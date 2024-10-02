import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import SqueezeNet1_1_Weights

def HPA_EfficientNet_B0_Model():
    '''
    Builds a model for the Human Protein Atlas dataset using EfficientNet-B0.
    '''

    # Carregar o modelo EfficientNet-B0 pré-treinado
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')

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

        # Inicializar os pesos do quarto canal com valores aleatórios entre -1 e 1
        new_conv_layer.weight[:, 3:, :, :] = 2 * torch.rand_like(first_conv_layer.weight[:, :1, :, :]) - 1

    # Substituir a camada original pela nova
    model.features[0][0] = new_conv_layer

    # Modificar a última camada para classificação multirrótulo
    num_labels = 19
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_labels)

    # Aplicar a função sigmoid na saída para multirrótulo
    model.add_module('sigmoid', torch.nn.Sigmoid())

    return model


class SqueezeNetCAM(nn.Module):
    def __init__(self, num_classes=19):
        super(SqueezeNetCAM, self).__init__()
        # Carregar o modelo SqueezeNet pré-treinado
        self.squeezenet = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

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

    def forward(self, x):
        # Extrair features da última camada convolucional
        features = self.squeezenet.features(x)

        # Aplicar a camada de classificação para obter os mapas de ativação
        output = self.squeezenet.classifier(features)

        # Aplicar global average pooling para transformar o output em [batch_size, num_classes]
        pooled_output = self.global_avg_pool(output)
        pooled_output = pooled_output.view(pooled_output.size(0), -1)

        return output, features, pooled_output
