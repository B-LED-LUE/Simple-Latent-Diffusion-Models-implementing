import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

device = "cuda" if torch.cuda.is_available() else "cpu"
setting = transforms.Compose([
    transforms.ToTensor(),
])

all_data = datasets.MNIST(root='./data', train=True, download=True, transform=setting)

five_idx = (all_data.targets == 5).nonzero(as_tuple=True)[0]
train_loader = DataLoader(Subset(all_data, five_idx), batch_size=128, shuffle=True)
train_loader_all = DataLoader(all_data, batch_size=128, shuffle=True)