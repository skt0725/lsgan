import os
from re import M
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from time import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# resize to 64 (32 too small)

# hyperparameter setting
data_dir = "./data"
batch_size = 64
latent_size = 100
total_epoch = 50
learning_rate = 0.0002
train_data = datasets.CIFAR10(root = data_dir, train = True, transform = transforms, download = True)
dataloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers = 4)

images, labels = next(iter(dataloader))
img = images[0].squeeze()
image = torch.permute(images[0].squeeze(), (1,2,0))
label = labels[0]
# plt.imshow(image)
# plt.title(label = label)
# plt.show()

class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.conv1= nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, 4, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        output = self.conv1(z)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(inplace = True)
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace = True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace = True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0)
        )
    def forward(self, z):
        output = self.conv1(z)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        return output.view(-1, 1)

discriminator = Discriminator().to(device)
generator = Generator(latent_size).to(device)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

criterion = nn.MSELoss().to(device)
average_time = 0
for epoch in range(total_epoch):
    start_time = time()
    for i, (image, label) in enumerate(dataloader):
        real_image = image.to(device)
        a = torch.zeros((image.size(0), 1)).to(device)
        b = torch.ones((image.size(0), 1)).to(device)
        c = torch.ones((image.size(0), 1)).to(device)
        # train discriminator
        dis_optimizer.zero_grad()
        real_output = discriminator(real_image)
        z = torch.randn((image.size(0), latent_size, 1, 1)).to(device)
        fake_image = generator(z)
        fake_output = discriminator(fake_image.detach())
        real_loss = criterion(real_output, b)/2
        fake_loss = criterion(fake_output, a)/2
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        dis_optimizer.step()

        # train generator
        gen_optimizer.zero_grad()
        z = torch.randn((image.size(0), latent_size, 1, 1)).to(device)
        fake_image = generator(z)
        fake_output = discriminator(fake_image)
        generator_loss = criterion(fake_output, c)/2

        generator_loss.backward()
        gen_optimizer.step()
    dir = f"./result"
    if not os.path.exists(dir):
        os.makedirs(dir)
    if epoch == 0:
        real_image = real_image.view(real_image.size(0), 3, 64, 64)
        save_image(real_image, "./result/real.png", normalize=True)
    if epoch % 10 == 0:
        fake_image = fake_image.view(fake_image.size(0), 3, 64, 64)
        # dir = f"./result/detach"
        save_image(fake_image, os.path.join(dir, f"{epoch}.png"), normalize=True)
    t = time()-start_time
    average_time += t
    print(f'Epoch {epoch}/{total_epoch} || discriminator loss={discriminator_loss:.4f}  || generator loss={generator_loss:.4f} || time {t:.3f}')
torch.save(discriminator.state_dict(), os.path.join(dir,"discriminator.ckpt"))
torch.save(generator.state_dict(), os.path.join(dir,"generator.ckpt"))
print(average_time/epoch)
