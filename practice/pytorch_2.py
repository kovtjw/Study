import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np

transf =tr.Compose([tr.Resize(8),tr.ToTensor()])
#1. 파이토치 제공 데이터 사용
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download=True, transform=transf)
testset = torchvision.datasets.CIFAR10(root = './data', train = False, download=True, transform=transf)
