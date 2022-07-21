import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from time import time
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_input = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# resize to 64 (32 too small)
'''
Comment: 
Yes, it is small. But when people report results using CIFAR10,
they keep the size since it is good for fair comparison (and for reproduce).
Definitely, the performance of GAN is affected by the size of sample image.
If you changed the size due to hard visualization, it would be better to upsample it after sampling, rather than to generate large image.
'''
'''
I resized for two reasons; 1. I wanted to increase the number of convolution layers 2. hardship of visualization
I understood the convention of using the original size for comparison, so I'll try with 32*32!
'''
# hyperparameter setting
data_dir = "./data"
batch_size = 64
latent_size = 100
total_epoch = 200
'''
Comment:
Little bit small number of epochs.
You may increase the number of epochs upto 200 for further convergence.
It would take some time, but it is common to wait several hours for training.
'''
'''
Set number of epochs to 200!
'''

learning_rate = 0.0002
train_data = datasets.CIFAR10(root = data_dir, train = True, transform = transform_input, download = True)
dataloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)

images, labels = next(iter(dataloader))
img = images[0].squeeze()
image = torch.permute(images[0].squeeze(), (1,2,0))
label = labels[0]
# plt.imshow(image)
# plt.title(label = label)
# plt.show()

'''
Comment:
Good, I recommend to see below repo which got many stars.
Both in the lsgan paper and the repo, generator contains a linear layer
to map noise vector (z) into higher dimension latent space.
repo: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/lsgan/lsgan.py
'''
'''
I tried to replace fc layer to deconvolutional layer at first.
Mapped noise vector from 100 to 1024 dimension using fc (non-linearity not introduced)
Further experiment : introduced non-linearity (maybe ReLU)
'''
'''
problem : checkerboard artifact due to deconvolution layer
breaking it down to interpolation->convolution can ease this problem
'''
class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.fc = nn.Linear(100, 1024)
        self.conv1= nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 1),
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
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        output = self.fc(z)
        output = output.view(output.size(0), 1024, 1, 1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
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
            nn.Conv2d(256, 1, 4, 1, 0)
        )
    def forward(self, z):
        output = self.conv1(z)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return output.view(-1, 1)

discriminator = Discriminator().to(device)
generator = Generator(latent_size).to(device)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

criterion = nn.BCEWithLogitsLoss().to(device)
average_time = 0
summary = SummaryWriter('logs/')
for epoch in range(total_epoch):
    start_time = time()
    for i, (image, label) in enumerate(dataloader):
        real_image = image.to(device)

        '''
        Comment:
        Nice details. The paper also defined parameters called a,b,c.
        If you see the paper section 3.2.3, they set a=-1, b= 1, and c=0, 
        which becomes the same loss equations as the vanilla gan except BCE -> MSE.
        So, for easier implementation, you can just change the 'criterion' in your vanilla gan implementation.
        The lsgan was spotlighted because this simple change cause better results.
        '''
        '''
        Equation 8 in lsgan satisfies b-c=1 && b-a=2 but its semantic is unclear to me.
        Rather, equation 9 is intuitive and I thought this is almost the same with vanilla gan.
        I guess need some help understanding this part!
        '''
        '''
        Question solved! equation 9 has similar form with vanilla gan
        '''
        a = torch.zeros((image.size(0), 1)).to(device)
        b = torch.ones((image.size(0), 1)).to(device)
        c = torch.ones((image.size(0), 1)).to(device)
        # train discriminator
        dis_optimizer.zero_grad()
        real_output = discriminator(real_image)
        z = torch.randn((image.size(0), latent_size)).to(device)
        fake_image = generator(z)
        fake_output = discriminator(fake_image.detach())
        real_loss = criterion(real_output, b)/2
        fake_loss = criterion(fake_output, a)/2
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        dis_optimizer.step()

        # train generator
        gen_optimizer.zero_grad()
        z = torch.randn((image.size(0), latent_size)).to(device)
        fake_image = generator(z)
        fake_output = discriminator(fake_image)
        generator_loss = criterion(fake_output, c)

        generator_loss.backward()
        gen_optimizer.step()

    result_save_dir = f"./result/bce"
    '''
    Comment: 
    'dir' is a name of internal function of Python. 
    It would be better change the variable name.
    '''
    '''
    Oops. Fixed it!
    I'll have to give more detailed naming to avoid this problem!
    '''
    transform = transforms.Resize((64,64))
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    # if epoch == 0:
    #     real_image = real_image.view(real_image.size(0), 3, 32, 32)
    #     real_image = transform(real_image)
    #     save_image(real_image, "./result/real.png", normalize=True, nrow=8)
    #     fake_image = fake_image.view(fake_image.size(0), 3, 32, 32)
    #     fake_image = transform(fake_image)
    #     save_image(fake_image, "./result/fake.png", normalize=True, nrow=8)
    if (epoch+1) % 10 == 0:
        fake_image = fake_image.view(fake_image.size(0), 3, 32, 32)
        fake_image = transform(fake_image)
        save_image(fake_image, os.path.join(result_save_dir, f"{epoch}.png"), normalize=True, nrow=8)
    t = time()-start_time
    average_time += t
    print(f'Epoch {epoch}/{total_epoch} || discriminator loss={discriminator_loss:.4f}  || generator loss={generator_loss:.4f} || time {t:.3f}')
    summary.add_scalar("bce/dis", discriminator_loss, epoch)
    summary.add_scalar("bce/gen", generator_loss, epoch)
    summary.flush()
summary.close()
torch.save(discriminator.state_dict(), os.path.join(result_save_dir,"discriminator.pth"))
torch.save(generator.state_dict(), os.path.join(result_save_dir,"generator.pth"))
'''
Comment:
A Pytorch convection for checkpoint path is using either '.pt' or '.pth' extension.
ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html
'''
'''
.ckpt is for tensorflow checkpoint path!
Thank you for pointing out. Fixed!
'''
print(average_time/epoch)


'''
Comment:
Well done! 

1) Generated image quality is not bad for the lsgan. 
I think there is a room for improvement if you change some implementations, but this is enough.

2) If you want to check if the lsgan is better than vanilla gan, you can train the vanilla gan using CIFAR10 and compare the result the lsgan.
Be aware to make the fair comparison (Ex. same model architecture, optimizer, batch_size, lr ...).
Obviously, the lsgan will be better than the vanilla gan, and you can see why the lsgan paper said that it is stable.
(Maybe, the word 'stable' means that it is more stable than the vanilla gan.)

3) Next step is wgan, which of background is harder than previous ones.
You don't need to understand everything in the paper.
At first time, just look around the concept of the paper.
(What is the problem the paper claims / How they prove that the claim is true / How they solve the problem.)
'''

'''
improvements
- use argparse for options
'''