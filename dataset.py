import os

import typing as tp
import pandas as pd

import torch
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image


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
    '''
    A PyTorch Dataset for the Human Protein Atlas dataset.
    '''
    def __init__(
        self,
        images_dir: str,
        labels_csv: str,
        indices: tp.Optional[tp.List[int]] = None,
        transform: tp.Optional[transforms.Compose] = None,
    ):
        '''
        Initializes the HPADataset object.
        Parameters:
            images_dir: str
                The directory where the images are stored.
            labels_csv: str
                Path to the CSV file containing labels.
            indices: Optional[List[int]]
                Indices to use for this subset of the dataset.
            transform: Optional[transforms.Compose]
                If not None, the images will be transformed.
        '''
        # Salvar os argumentos
        self.images_dir = images_dir
        self.transform = transform

        # Listar todos os arquivos no diretório
        self.filenames = os.listdir(self.images_dir)

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
        images = [read_image(os.path.join(self.images_dir, self.filenames[idx] + color + ".png")) for color in colors]
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


class HPADatasetFourChannelsImages(HPADataset):
    '''
    A PyTorch Dataset for the Human Protein Atlas dataset.
    '''
    def __init__(
        self,
        images_dir: str,
        labels_csv: str,
        indices: tp.Optional[tp.List[int]] = None,
        transform: tp.Optional[transforms.Compose] = None,
    ):
        '''
        Initializes the HPADataset object.
        Parameters:
            images_dir: str
                The directory where the images are stored.
            labels_csv: str
                Path to the CSV file containing labels.
            indices: Optional[List[int]]
                Indices to use for this subset of the dataset.
            transform: Optional[transforms.Compose]
                If not None, the images will be transformed.
        '''
        # Salvar os argumentos
        self.images_dir = images_dir
        self.transform = transform

        # Listar todos os arquivos no diretório
        self.filenames = os.listdir(self.images_dir)

        # Remover a extensão dos arquivos
        self.filenames = [filename.split(".")[0] for filename in self.filenames]

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

        # Pegar imagem
        image = read_image(os.path.join(self.images_dir, self.filenames[idx]) + ".png")
        
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
