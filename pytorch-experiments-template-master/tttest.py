from __future__ import print_function, division
from torchvision import datasets

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from useless.Resize import *

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode
data_transform = transforms.Compose([
    Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root='E:\pet\pytorch-experiments-template-master\data', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

test_dataset = datasets.ImageFolder(root='E:\pet\pytorch-experiments-template-master\data', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
