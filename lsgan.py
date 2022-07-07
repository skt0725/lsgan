import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# hyperparameter setting
data_dir = "./data"
batch_size = 128
train_data = datasets.CIFAR10(root = data_dir, train = True, transform = transforms, download = True)
dataloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers = 4)

images, labels = next(iter(dataloader))
image = torch.permute(images[0].squeeze(), (1,2,0))
label = labels[0]
plt.imshow(image)
plt.title(label = label)
plt.show()