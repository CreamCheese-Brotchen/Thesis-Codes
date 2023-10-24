import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
from torch import nn
from torch.nn import functional as F  
import matplotlib.pyplot as plt
import torchvision
from torch import nn
import argparse
# from augmentation_folder.dataset_loader import IndexDataset, create_dataloaders, model_numClasses
from dataset_loader import IndexDataset, create_dataloaders, model_numClasses
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import os



class VAE(nn.Module):
    def __init__(self, image_size, channel_num, kernel_num, z_size, loss_func):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.loss_func = loss_func 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # encoder
        if self.image_size == 32:
          self.encoder = nn.Sequential(
              self._conv(channel_num, kernel_num // 4),
              self._conv(kernel_num // 4, kernel_num // 2),
              self._conv(kernel_num // 2, kernel_num),
          )
        elif self.image_size == 256:
          self.encoder == nn.Sequential(            
            self._conv(channel_num, kernel_num // 8),
            self._conv(kernel_num // 8, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num)
            )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode x
        encoded = self.encoder(x.to(self.device))

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the reconstructed image.
        return (mean, logvar), x_reconstructed

    # intergrate the latent dim from encoder with GANs model
    def get_latent(self, x):
        
        encoded = self.encoder(x)
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)  # z.shape = (batch_size, 128)
        # z_projected = self.project(z).view(
        #     -1, self.kernel_num,
        #     self.feature_size,
        #     self.feature_size,
        # )
        return z
    
    def get_singleImg(self, x):
        self.decoder.eval()
        self.encoder.eval()
        with torch.no_grad():
            _, singleImg = self.forward(x.unsqueeze(0))
        return singleImg
    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size()))
        ).to(self.device)
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        if self.loss_func:
            loss = self.loss_func(x_reconstructed, x)
        else:
            # print('using default loss function for vae')
            loss = nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)
            # loss = nn.BCELoss(size_average=True)(x_reconstructed, x) 
            # loss = nn.BCELoss(reduction='mean')(x_reconstructed, x)
            # print('loss', loss)
            # loss = F.mse_loss(x_reconstructed, x)

        return loss

    def kl_divergence_loss(self, mean, logvar):
        kld_loss_new = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)
        # kl_original = ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()

        # print('k1_loss', kld_loss_new)
        # print('kl_original', kl_original)
        return kld_loss_new


    def sample(self, size):
        z = Variable(
            torch.randn(size, self.z_size)
        ).to(self.device)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data


    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)


 
def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch

def train_model(model, data_loader, epochs=10, lr=3e-04, weight_decay=1e-5, tensorboard_comment='test_run'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(comment=tensorboard_comment)        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    model.train()
    for epoch in range(epochs):
        for i, (x, _, _) in enumerate(data_loader['train']):
            x = Variable(x).to(device)
            # flush gradients and run the model forward
            optimizer.zero_grad()
            (mean, logvar), x_reconstructed = model(x)
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            # print('reconstruction_loss.shape: ', reconstruction_loss)
            kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
            total_loss = reconstruction_loss + kl_divergence_loss

            # backprop gradients from the loss
            total_loss.backward()
            optimizer.step()
        
        ###### 
        original_img = x[-1].unsqueeze(0)
        reconstructed_img = x_reconstructed[-1].unsqueeze(0).detach()
        stacked_images = torch.cat([original_img, reconstructed_img])
        writer.add_scalar('Vae/total loss', total_loss.detach().cpu().numpy(), epoch+1)
        writer.add_images('Vae/orig vs x_recons imgs ', stacked_images, epoch+1)

        print('epoch ', epoch,
              ", recons_loss:", reconstruction_loss.detach().cpu().numpy(),
              ", k1_loss:", kl_divergence_loss.detach().cpu().numpy(),
              ", total loss:", total_loss.detach().cpu().numpy())
        
    writer.close()
        
        # save the checkpoint.
        # save_checkpoint(model, checkpoint_dir, epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resnet Training script')

    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=("MNIST", "CIFAR10", "FashionMNIST", "SVHN", "Flowers102", "Food101"), help='Dataset name')
    parser.add_argument('--run_epochs', type=int, default=5, help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (default: 64)')
    parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')
    parser.add_argument('--tensorboard_comment', type=str, default='vae test_run', help='Comment to append to tensorboard logs')
    parser.add_argument("--kernel_num", type=int, default=128, help="Number of kernels in the first layer of the VAE")
    parser.add_argument("--z_size", type=int, default=128, help="Size of the latent vector")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--loss_func", default=False, help="Flag to use BCELoss for testing")  # not given loss_func, use original lossFunc 
    args = parser.parse_args()
    print(f"Script Arguments: {args}", flush=True)

    if args.dataset in ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN']:
        transformSize = transforms.Compose([
            transforms.transforms.ToTensor(),
        ])
    elif args.dataset in ['Flowers102', 'Food101']:
        transformsSize = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.transforms.ToTensor(),
        ])
    dataset_loader = create_dataloaders(transformSize, transformSize, args.batch_size, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)
    num_channel = dataset_loader['train'].dataset[0][0].shape[0]
    image_size = dataset_loader['train'].dataset[0][0].shape[1]
    vae = VAE(
        image_size=image_size,
        channel_num=num_channel,
        kernel_num=args.kernel_num,
        z_size=args.z_size,
        loss_func=args.loss_func,
    )
    train_model(vae, dataset_loader,
            epochs=args.run_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            tensorboard_comment = args.tensorboard_comment,
            )
    
    # ##############################
    # ## GANs
    # ##############################
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if isinstance(args.gans_latentDim, int):
    #     gans_latentVector = args.gans_latentDim
    # else:
    #     print("using the latent vector from the VAE")
    #     # gans_latentVector = 
    # netD = Discriminator(in_channels=num_channel, image_size=image_size).to(device)
    # netG = Generator(channel_num=num_channel, input_size=image_size, input_dim=gans_latentVector).to(device)
    # netD.apply(weights_init)
    # netG.apply(weights_init)
    # trainer = gans_trainer(netD=netD, netG=netG, dataloader=dataset_loader, num_channel=num_channel, input_size=image_size, latent_dim=100,
    #                        num_epochs=args.run_epochs, batch_size=args.batch_size, lr=args.lr, criterion=nn.BCELoss(),
    #                        tensorboard_comment=args.tensorboard_comment)
    # trainer.training_steps()


