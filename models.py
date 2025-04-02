import torch
import torch.nn as nn
import timm


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

        # Print the model architecture
        # print(self.model)


    def load_from_checkpoint(self, weights):
        """
        Load pretrained weights from a file.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(weights, map_location=device)
        self.model.load_state_dict(state_dict)

        print("Weights loaded successfully!")


    def forward(self, x):
        return self.model(x)
