import torch.nn as nn
import torch
import dataset_loader
from dataset_loader import create_dataloaders
from torchvision import transforms
import torchvision.utils as vutils
import argparse
from torch.nn import functional as F
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import tensorboard



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, in_channels, image_size, alpha=0.2):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.alpha = alpha

        self.features = image_size // 32
        
        if self.image_size==256:
          self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )        
        elif self.image_size==32:
          self.model = nn.Sequential(
              nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(0.2, inplace=True),

              nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
              nn.BatchNorm2d(128),
              nn.LeakyReLU(0.2, inplace=True),

              nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
              nn.BatchNorm2d(256),
              nn.LeakyReLU(0.2, inplace=True),

              nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
              nn.BatchNorm2d(512),
              nn.LeakyReLU(0.2, inplace=True),

              nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
              nn.BatchNorm2d(1024),
              nn.LeakyReLU(0.2, inplace=True)
          )
       
        self.output_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024 * self.features * self.features, 1),
                nn.Sigmoid())


    def forward(self, x):
        x = self.model(x)
        output = self.output_layer(x)
        return output


class Generator(nn.Module):
    def __init__(self, channel_num, input_size, input_dim=100):
        super(Generator, self).__init__()
        self.channel_num = channel_num
        self.input_size = input_size
        self.input_dim = input_dim

        layers = [
            nn.ConvTranspose2d(self.input_dim, 512, kernel_size=4, stride=1, padding=0),  # 1st
            nn.BatchNorm2d(512),
            nn.ReLU(), 
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),        # 2nd
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),        # 3rd
            nn.BatchNorm2d(128),
            nn.ReLU()]

        if self.input_size == 32: 
            layers += [
              nn.ConvTranspose2d(128, self.channel_num, kernel_size=4, stride=2, padding=1),  # 4th
              nn.Tanh()
        ]
        elif self.input_size == 256:
            layers += [
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4th
                nn.Tanh(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
                nn.ConvTranspose2d(16, self.channel_num, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            ]
        
        self.model = nn.Sequential(*layers)
     

    def forward(self, x):
        x = self.model(x)
        return x




class gans_trainer():
    def __init__(self, netD, netG,  dataloader, num_channel, input_size, num_epochs, batch_size, lr, criterion, tensorboard_comment, latent_dim=100):
        self.netD = netD
        self.netG = netG
        self.dataloader = dataloader
        self.num_channel = num_channel
        self.input_size = input_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(self.batch_size, self.latent_dim, 1, 1)  #是否是用fixed_noise, 或者是每个epoch都用不同的noise
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensorboard_comment = tensorboard_comment
        self.fixed_noise = torch.randn(1, self.latent_dim, 1, 1, device=self.device) # to viszualize the training process with one img

    def training_steps(self):
        print("Starting Training Loop...")

        real_label = 1.
        fake_label = 0.

        writer = SummaryWriter(comment=self.tensorboard_comment)
        
        for epoch in range(self.num_epochs):

            G_losses = []
            D_losses = []    

            for i, (img_tensor, label_tensor, id) in enumerate(self.dataloader['train']):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # real_cpu = data[0].to(device)
                real_cpu = img_tensor.to(self.device)
                b_size = real_cpu.size(0)       # batch_size
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D 
                output = self.netD(real_cpu).view(-1)   # input with real img and get output: pred_probability with batch_size
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)  
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors of latent_dim as input for G
                noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)   # fake_img
                label.fill_(fake_label)   # fake labels
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = errD_fake.mean().item()   # avg_loss_net
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = errG.mean().item()  # the original: output.mean().item()
                # Update G
                self.optimizerG.step()

            # Save Losses for plotting later 
            with torch.no_grad():
                fakeImg_training = self.netG(self.fixed_noise).detach().cpu()
                writer.add_scalar('loss_D', D_G_z1, epoch+1)
                writer.add_scalar('loss_G', D_G_z2, epoch+1)
                writer.add_image('fake_images', fakeImg_training[-1], epoch+1)

            writer.close()

            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, self.num_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # finalEpoch_fakeImg = self.netG(self.fixed_noise).detach().cpu()

    def customed_generator(self, customed_latent):
        self.netG.eval()
        generated_imags = self.netG(customed_latent)
        return generated_imags

    # def get_fake_images(self, image_loader):
        
    #     noise = torch.randn(num_images, self.latent_dim, 1, 1, device=self.device)
    #     fake_imgs = self.netG(noise)
    #     return fake_imgs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resnet Training script')

    parser.add_argument('--dataset', type=str, default='MNIST', choices=("MNIST", "CIFAR10", "FashionMNIST", "SVHN", "Flowers102", "Food101"), help='Dataset name')
    parser.add_argument('--run_epochs', type=int, default=5, help='Number of epochs to run')
    parser.add_argument('--tensorboard_comment', type=str, default='debug', help='Comment to append to tensorboard logs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')

    args = parser.parse_args()
    print(f"Script Arguments: {args}", flush=True)

    if args.dataset in ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN']:
        transformSize = transforms.Compose([
            transforms.transforms.ToTensor(),
        ])
    elif args.dataset in ['Flowers102', 'Food101']:
        transformSize = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.transforms.ToTensor(),
        ])

    dataset_loaders = create_dataloaders(transformSize, transformSize, args.batch_size, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)
    num_channel = dataset_loaders['train'].dataset[0][0].shape[0]
    image_size = dataset_loaders['train'].dataset[0][0].shape[1]
    latent_dim = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netD = Discriminator(in_channels=num_channel, image_size=image_size).to(device)
    netG = Generator(channel_num=num_channel, input_size=image_size, input_dim=latent_dim).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)
    trainer = gans_trainer(netD=netD, netG=netG, dataloader=dataset_loaders, num_channel=num_channel, input_size=image_size, latent_dim=100,
                           num_epochs=args.run_epochs, batch_size=args.batch_size, lr=args.lr, criterion=nn.BCELoss(),
                           tensorboard_comment=args.tensorboard_comment)
    trainer.training_steps()