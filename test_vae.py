import numpy as np
import matplotlib
import pandas as pd
import torch
from torchvision import transforms
import torchmetrics
from pytorch_lightning import LightningModule, Trainer, seed_everything
from dataset_loader import IndexDataset, create_dataloaders, model_numClasses 
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import os
from torch import nn
from torch.nn import functional as F
from torch import optim
from VAE_model import VAE, train_model
from GANs_model import gans_trainer, Discriminator, Generator, weights_init
import torch.nn as nn
from torchvision import transforms
import argparse
from torch.nn import functional as F
import tensorboard

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE & GANs Training script')
    
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=("MNIST", "CIFAR10", "FashionMNIST", "SVHN", "Flowers102", "Food101"), help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (default: 64)')
    parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')

    # vae
    parser.add_argument('--vae_runEpochs', type=int, default=5, help='Number of epochs to run')
    parser.add_argument("--vae_lossFunc", default=False, help="Flag to use BCELoss for testing")
    parser.add_argument('--vae_tensorboard_comment', type=str, default='vae test_run', help='Comment to append to tensorboard logs')
    parser.add_argument('--vae_lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--kernel_num", type=int, default=128, help="Number of kernels in the first layer of the VAE")
    parser.add_argument("--z_size", type=int, default=128, help="Size of the latent vector")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")

    parser.add_argument("--gans_latentDim", default=100, help="Size of the latent vector for the GANs")  # if use "vae", input with vae
    parser.add_argument('--gans_runEpochs', type=int, default=5, help='Number of epochs to run')
    parser.add_argument('--gans_lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gans_tensorboard_comment', type=str, default='gans test_run', help='Comment to append to tensorboard logs')
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

        
    ##############################
    ## VAE
    ##############################
    vae = VAE(
        image_size=image_size,
        channel_num=num_channel,
        kernel_num=args.kernel_num,
        z_size=args.z_size,
        loss_func=args.vae_lossFunc,  # if 不给loss_func, 默认使用nn.BCELoss(size=False)/x.size(0) 
    )
    print("VAE starts training")
    train_model(vae, dataset_loader,
            epochs=args.vae_runEpochs,
            lr=args.vae_lr,
            weight_decay=args.weight_decay,
            tensorboard_comment=args.vae_tensorboard_comment,
        )
    
    ##############################
    ## GANs
    ##############################
    print("GANs starts training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(args.gans_latentDim, int):
        GANs_latentDim = args.gans_latentDim
        print("using the latent_dim Param from the args:", GANs_latentDim)
    else:
        batch_images, _, _ = next(iter(dataset_loader['train']))
        temp_singleImg = batch_images[0].unsqueeze(0).to(device)
        GANs_latentDim = len(vae.get_latent(temp_singleImg)[0].view(1, -1).squeeze(0))  # c.a. 2048 (128*4*4)
        
        print("using the latent_dim Param from the trained VAE:", GANs_latentDim)
    netD = Discriminator(in_channels=num_channel, image_size=image_size).to(device)
    netG = Generator(channel_num=num_channel, input_size=image_size, input_dim=GANs_latentDim).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)
    trainer = gans_trainer(netD=netD, netG=netG, dataloader=dataset_loader, num_channel=num_channel, input_size=image_size, latent_dim=GANs_latentDim,
                           num_epochs=args.gans_runEpochs, batch_size=args.batch_size, lr=args.gans_lr, criterion=nn.BCELoss(),
                           tensorboard_comment=args.gans_tensorboard_comment)
    trainer.training_steps()

    if not isinstance(args.gans_latentDim, int):
        gans_vaeLatent_writer= SummaryWriter(comment="using vae latent for generating imgs with GANs")
        batch_vaeLatent = vae.get_latent(batch_images.to(device))  #.view(2, -1)  # batch_vaeLatent.shape = (3, 128*4*4)
        new_size = (batch_vaeLatent.size(0), -1, 1, 1)
        batch_vaeLatent = batch_vaeLatent.view(new_size)
        result = trainer.get_imgs(batch_vaeLatent)  # input.shape = (batch_size, 128*4*4, 1, 1), output.shape = (batch_size, 3, 32, 32)
        combine_imgs = torch.cat((batch_images[:8], result[:8]), 0)
        gans_vaeLatent_writer.add_images("original vs vaeLatent_GANs_imgs", combine_imgs, dataformats="NCHW", global_step=0)
        gans_vaeLatent_writer.close()






  

  

