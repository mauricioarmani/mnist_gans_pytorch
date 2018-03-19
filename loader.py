import cPickle as pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = datasets.MNIST('dataset/', train=True, transform=transform)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data