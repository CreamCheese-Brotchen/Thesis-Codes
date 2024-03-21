import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.callbacks import SRImageLoggerCallback
from pl_bolts.datamodules import TVTDataModule
from pl_bolts.datasets.utils import prepare_sr_datasets
from pl_bolts.models.gans.srgan.components import SRGANDiscriminator, SRGANGenerator, VGG19FeatureExtractor
from torchvision.models import vgg19
from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import argparse
from torchvision.transforms import ToTensor
from torchvision import transforms
# from augmentation_folder.dataset_loader import create_dataloaders
from dataset_loader import create_dataloaders
from PIL import Image



class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class
    Values
        channels: the number of channels throughout the residual block, a scalar
    '''

    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        base_channels: number of channels throughout the generator, a scalar
        n_ps_blocks: number of PixelShuffle blocks, a scalar
        n_res_blocks: number of residual blocks, a scalar
    '''

    def __init__(self, base_channels=64, n_ps_blocks=2, n_res_blocks=16):
        super().__init__()
        # Input layer
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_res_blocks):
            res_blocks += [ResidualBlock(base_channels)]

        res_blocks += [
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # PixelShuffle blocks
        ps_blocks = []
        for _ in range(n_ps_blocks):
            ps_blocks += [
                nn.Conv2d(base_channels, 4 * base_channels, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        self.ps_blocks = nn.Sequential(*ps_blocks)

        # Output layer
        self.out_layer = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        x_res = self.in_layer(x)
        x = x_res + self.res_blocks(x_res)
        x = self.ps_blocks(x)
        x = self.out_layer(x)
        return x
   
   
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        base_channels: number of channels in first convolutional layer, a scalar
        n_blocks: number of convolutional blocks, a scalar
    '''

    def __init__(self, base_channels=64, n_blocks=3):
        super().__init__()
        self.blocks = [
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        cur_channels = base_channels
        for i in range(n_blocks):
            self.blocks += [
                nn.Conv2d(cur_channels, 2 * cur_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(2 * cur_channels),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(2 * cur_channels, 2 * cur_channels, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(2 * cur_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            cur_channels *= 2

        self.blocks += [
            # You can replicate nn.Linear with pointwise nn.Conv2d
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cur_channels, 2 * cur_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * cur_channels, 1, kernel_size=1, padding=0),

            # Apply sigmoid if necessary in loss function for stability
            nn.Flatten(),
        ]

        self.layers = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.layers(x)
    


class Loss(nn.Module):
    '''
    Loss Class
    Implements composite content+adversarial loss for SRGAN
    Values:
        device: 'cuda' or 'cpu' hardware to put VGG network on, a string
    '''

    def __init__(self, device='cuda'):
        super().__init__()

        vgg = vgg19(pretrained=True).to(device)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    @staticmethod
    def img_loss(x_real, x_fake):
        return F.mse_loss(x_real, x_fake)

    def adv_loss(self, x, is_real):
        target = torch.zeros_like(x) if is_real else torch.ones_like(x)
        return F.binary_cross_entropy_with_logits(x, target)

    def vgg_loss(self, x_real, x_fake):
        return F.mse_loss(self.vgg(x_real), self.vgg(x_fake))

    def forward(self, generator, discriminator, hr_real, lr_real):
        ''' Performs forward pass and returns total losses for G and D '''
        hr_fake = generator(lr_real)
        fake_preds_for_g = discriminator(hr_fake)
        fake_preds_for_d = discriminator(hr_fake.detach())
        real_preds_for_d = discriminator(hr_real.detach())

        g_loss = (
            0.001 * self.adv_loss(fake_preds_for_g, False) + \
            0.006 * self.vgg_loss(hr_real, hr_fake) + \
            self.img_loss(hr_real, hr_fake)
        )
        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) + \
            self.adv_loss(fake_preds_for_d, False)
        )

        return g_loss, d_loss, hr_fake
    

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

def show_tensor_images(image_tensor):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:4], nrow=4)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def train_srresnet(srresnet, dataloader, device, lr=1e-4, run_epochs=10, tensorboard_comment='test train_srresnet'):

    srresnet = srresnet.to(device).train()
    optimizer = torch.optim.Adam(srresnet.parameters(), lr=lr)
    writer = SummaryWriter(comment=tensorboard_comment)

    # total_dataset_size = len(dataloader.dataset)
    # batch_size = dataloader.batch_size
    # num_batches = (total_dataset_size + batch_size - 1) // batch_size


    # while cur_step < total_steps:
    for epoch in range(run_epochs):
        total_loss_epoch = 0
        for hr_real, lr_real in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)

            
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    hr_fake = srresnet(lr_real)
                    loss = Loss.img_loss(hr_real, hr_fake)
            else:
                hr_fake = srresnet(lr_real)
                loss = Loss.img_loss(hr_real, hr_fake)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()

        writer.add_image('Img/lr_real', (lr_real[-1]*2-1).detach().cpu(), epoch+1)
        writer.add_image('Img/hr_fake', hr_fake[-1].to(hr_real.dtype).detach().cpu(), epoch+1)
        writer.add_image('Img/hr_real', hr_real[-1].detach().cpu(), epoch+1)
        writer.add_scalar('Loss', total_loss_epoch/len(dataloader), epoch+1)

    writer.close()    



def train_srgan(generator, discriminator, dataloader, device, lr=1e-4, run_epochs=5, tensorboard_comment='test train_srgan'):
    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()
    loss_fn = Loss(device=device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lambda _: 0.1)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lambda _: 0.1)

    lr_step = run_epochs // 2
    cur_step = 0

    writer = SummaryWriter(comment=tensorboard_comment)

    # while cur_step < total_steps:
    for epoch in range(run_epochs):

        total_g_loss_epoch = 0 
        total_d_loss_epoch = 0

        for hr_real, lr_real in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)

            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    g_loss, d_loss, hr_fake = loss_fn(
                        generator, discriminator, hr_real, lr_real,
                    )
            else:
                g_loss, d_loss, hr_fake = loss_fn(
                    generator, discriminator, hr_real, lr_real,
                )

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # mean_g_loss += g_loss.item() / display_step
            # mean_d_loss += d_loss.item() / display_step

            total_g_loss_epoch += g_loss.item()
            total_d_loss_epoch += d_loss.item()

            if cur_step == lr_step:
                g_scheduler.step()
                d_scheduler.step()
                print('Decayed learning rate by 10x.')

            # if cur_step % display_step == 0 and cur_step > 0:
            #     # print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'.format(cur_step, mean_g_loss, mean_d_loss))
            #     show_tensor_images(lr_real * 2 - 1)
            #     show_tensor_images(hr_fake.to(hr_real.dtype))
            #     show_tensor_images(hr_real)
            #     mean_g_loss = 0.0
            #     mean_d_loss = 0.0

            cur_step += 1

        writer.add_image('Img/lr_real', (lr_real[-1]*2-1).detach().cpu(), epoch+1)
        writer.add_image('Img/hr_fake', hr_fake[-1].to(hr_real.dtype).detach().cpu(), epoch+1)
        writer.add_image('Img/hr_real', hr_real[-1].detach().cpu(), epoch+1)
        writer.add_scalar('Loss/g_loss', total_g_loss_epoch/len(dataloader), epoch+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='srgan Training script')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--reduce_dataset', action='store_true', help='Reduce dataset size')
    parser.add_argument('--srgan_lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--srgan_epochs', type=int, default=5, help='Total number of training steps')
    parser.add_argument('--srgan_tensorboardComment', type=str, default='test srgan', help='Tensorboard comment')

    args = parser.parse_args()
    print(f"Script Arguments: {args}", flush=True)
    if args.dataset in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10']:
        mean = (0.5,)
        std = (0.5, 0.5, 0.5) 
        transforms_smallSize = transforms.Compose([
        # transforms.Resize((32, 32), interpolation=Image.BICUBIC),
        transforms.transforms.ToTensor(),
        # transforms.Normalize(mean, meanP,
        ])
        dataset_loaders = create_dataloaders(transforms_smallSize, transforms_smallSize, args.batch_size, args.dataset, add_idx=False, reduce_dataset=args.reduce_dataset, srgan=True)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transforms_largSize= transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.transforms.ToTensor(),
        transforms.Normalize(mean, std),])
        dataset_loaders = create_dataloaders(transforms_largSize, transforms_largSize, args.batch_size, args.dataset, add_idx=False, reduce_dataset=args.reduce_dataset, srgan=True)
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator(n_res_blocks=16, n_ps_blocks=2)
    train_srresnet(generator, dataset_loaders['train'], device, lr=args.srgan_lr, run_epochs=args.srgan_epochs, tensorboard_comment= args.srgan_tensorboardComment)
    torch.save(generator, 'GANs_folder\save_trained_model\srresnet.pt')

    generator = torch.load('GANs_folder\save_trained_model\srresnet.pt')
    discriminator = Discriminator(n_blocks=1, base_channels=8)
    train_srgan(generator, discriminator, dataset_loaders['train'], device, lr=args.srgan_lr, run_epochs=args.srgan_epochs)
    torch.save(generator, 'GANs_folder\save_trained_model\srgenerator.pt')
    torch.save(discriminator, 'GANs_folder\save_trained_model\srdiscriminator.pt')
