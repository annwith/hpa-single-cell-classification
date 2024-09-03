import os

import typing as tp
import numpy as np
import pandas as pd

import torch
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image

from sklearn.model_selection import train_test_split


classes_map = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Intermediate filaments",
    9: "Actin filaments",
    10: "Microtubules",
    11: "Mitotic spindle",
    12: "Centrosome",
    13: "Plasma membrane",
    14: "Mitochondria",
    15: "Aggresome",
    16: "Cytosol",
    17: "Vesicles and punctate cytosolic patterns",
    18: "Negative",
}


class HPADataset():
    def __init__(
        self, 
        dir: str,
        labels_csv: str,
        indices: tp.Optional[tp.List[int]] = None,
        transform: tp.Optional[transforms.Compose] = None,
    ):
        '''
        Initializes the ImageDataset object.
        Parameters:
            dir: str
                The directory where the images are stored.
            labels_csv: str
                Path to the CSV file containing labels.
            indices: Optional[List[int]]
                Indices to use for this subset of the dataset.
            transform: Optional[transforms.Compose]
                If not None, the images will be transformed.
        '''
        # Salvar os argumentos
        self.dir = dir
        self.transform = transform

        # Listar todos os arquivos no diretório
        self.filenames = os.listdir(self.dir)

        # Extrair a parte do nome do arquivo antes do primeiro "_"
        self.filenames = [filename.split("_")[0] for filename in self.filenames]

        # Remover duplicatas
        self.filenames = list(set(self.filenames))

        # Ordenar a lista de arquivos
        self.filenames.sort()

        # Carregar o arquivo CSV com os rótulos
        self.labels = pd.read_csv(labels_csv).set_index("ID")

        # Ordenar os rótulos de acordo com a ordem dos arquivos
        self.labels = self.labels.loc[self.filenames]

        # Converter a coluna 'Label' de strings para listas de inteiros
        self.labels['Label'] = self.labels['Label'].apply(lambda x: list(map(int, x.split('|'))))

        # Transformar os rótulos em binário
        self.binary_labels = torch.zeros((len(self.labels), 19), dtype=torch.float32)

        # Para cada rótulo, definir os índices correspondentes como 1
        for i, label_list in enumerate(self.labels['Label']):
            for label in label_list:
                self.binary_labels[i, label] = 1

        # Se índices são fornecidos, selecionar apenas esses índices
        if indices is not None:
            self.filenames = [self.filenames[i] for i in indices]
            self.binary_labels = self.binary_labels[indices]

    def __len__(self) -> int:
        '''
        Returns the number of images in the dataset.
        Returns:
            int
                The number of images in the dataset.
        '''
        return len(self.filenames)

    def __getitem__(self, idx) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns the image and its label.
        Parameters:
            idx: int
                The index of the image to be returned.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                A tuple containing the image and its label.
        '''
        # Checar se o índice é válido
        if idx >= self.__len__():
            raise IndexError
        
        # Pegar todas a imagens
        colors = ["_green", "_blue", "_red", "_yellow"]
        images = [read_image(os.path.join(self.dir, self.filenames[idx] + color + ".png")) for color in colors]
        image = torch.cat(images, 0)

        # Aplicar transformações
        if self.transform:
            image = self.transform(image)
        
        # Pegar os rótulos
        label = self.binary_labels[idx]

        return image, label
    
    def set_transform(self, transform: tp.Optional[transforms.Compose]) -> None:
        '''
        Sets the transform attribute.
        Parameters:
            transform: Optional[transforms.Compose]
                The transform to be set.
        '''
        self.transform = transform


def train_valid_split_multilabel(
    dataset_dir: str,
    labels_csv: str,
    train_transform: tp.Optional[transforms.Compose] = None,
    valid_transform: tp.Optional[transforms.Compose] = None,
    test_size=0.25
):
    # Carregar o dataset completo
    dataset = HPADataset(dir=dataset_dir, labels_csv=labels_csv)
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
        stratify=binary_labels.sum(axis=1)  # Estratificar baseado no número de rótulos por amostra
    )

    train_dataset = HPADataset(
        dir=dataset_dir, 
        labels_csv=labels_csv, 
        indices=train_indices, 
        transform=train_transform)
    valid_dataset = HPADataset(
        dir=dataset_dir, 
        labels_csv=labels_csv, 
        indices=valid_indices, 
        transform=valid_transform)

    return train_dataset, valid_dataset


def train_transformations() -> transforms.Compose:
    '''
    Returns a composition of transformations to be applied to the training images.
    Returns:
        transforms.Compose
            The composition of transformations.
    '''
    return transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToImage(), # Transformar de tensor para imagem
    transforms.ToDtype(torch.float32, scale=True)  
])


def valid_transformations() -> transforms.Compose:
    '''
    Returns a composition of transformations to be applied to the validation images.
    Returns:
        transforms.Compose
            The composition of transformations.
    '''
    return transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToImage(), # Transformar de tensor para imagem
    transforms.ToDtype(torch.float32, scale=True)  
])
