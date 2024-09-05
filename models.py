import torch
import torch.nn as nn
from torchvision import models

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
