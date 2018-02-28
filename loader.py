import cPickle as pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class Data(Dataset):
    def __init__(self, X, Y):
       super(Data, self).__init__()
       self.X = X
       self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

    def __add__(self, other):
        return ConcatDataset([self, other])


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = datasets.MNIST('dataset/', train=True, transform=transform)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data