import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import torch
import torchvision
import argparse
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.utils.data as data_utils
from torchvision import transforms
import torchmetrics
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from more_itertools import flatten
import itertools
from collections import Counter
from pytorch_lightning import LightningModule, Trainer
import torch.nn as nn
import PIL
from torch.optim import lr_scheduler
from torchvision.transforms import InterpolationMode
from torch.nn import functional as F
# from memory_profiler import profile
# import sys 

from augmentation_folder.dataset_loader import IndexDataset, create_dataloaders, model_numClasses, boardWriter_generator
from augmentation_folder.augmentation_methods import simpleAugmentation_selection, AugmentedDataset, vae_augmentation, vae_gans_augmentation, DenoisingModel
from VAE_folder.VAE_model import VAE, train_model
from resnet_model import Resnet_trainer
from GANs_folder.GANs_model import Discriminator, Generator, gans_trainer, weights_init 
from param_tune_folder.lrSearch import lrSearch

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Resnet Training script')

  parser.add_argument('--dataset', type=str, default='CIFAR10', choices=("MNIST", "CIFAR10", 'CINIC10', "FashionMNIST", "SVHN", "Flowers102", "Food101"), help='Dataset name')
  parser.add_argument('--entropy_threshold', type=float, default=1.7, help='Entropy threshold')
  parser.add_argument('--run_epochs', type=int, default=5, help='Number of epochs to run')
  parser.add_argument('--candidate_start_epoch', type=int, default=0, help='Epoch to start selecting candidates. Candidate calculation begind after the mentioned epoch')
  parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
  parser.add_argument('--customLr_flag', action='store_true', help='Define the lr customly')
  parser.add_argument('--l2', type=float, default=1e-5, help='L2 regularization')
  parser.add_argument('--norm', action='store_true', help='Normalize the dataset')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
  parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')
  parser.add_argument('--pretrained_flag', action='store_true', help='Use pretrained model')
  parser.add_argument('--addComment', default=None, help='Aditional comment to tensorboard')
  parser.add_argument('--accumulation_steps', type=int, default=None, help='Number of accumulation steps')
  parser.add_argument('--lr_scheduler_flag', action='store_true', help='Use lr scheduler')
  parser.add_argument('--transfer_learning', action='store_true', help='Use transfer learning')
  parser.add_argument('--random_candidateSelection', action='store_true', help='Randomly select candidates')
  
  parser.add_argument('--augmentation_type', type=str, default=None, choices=("vae", "simple", 'simple_crop', 'simple_centerCrop', "GANs", "navie_denoiser", 'builtIn_denoiser', 'builtIn_vae'), help='Augmentation type')
  parser.add_argument('--simpleAugmentation_name', type=str, default=None, choices=("random_color", "center_crop", "gaussian_blur", "rotation",
                                                                                   "elastic_transform", "random_perspective", "random_resized_crop", 
                                                                                   "random_invert", "random_posterize", "rand_augment", "augmix"), help='Simple Augmentation name')
  parser.add_argument('--k_epoch_sampleSelection', type=int, default=3, help='Number of epochs to select the common candidates')
  parser.add_argument('--augmente_epochs_list', type=list, default=None, help='certain epoch to augmente the dataset')
  parser.add_argument('--AugmentedDataset_func', type=int, default=1, choices=(1,2,3), help='Choose the way to replace the original image with augmented img temporily or set')


  parser.add_argument('--residualConnection_flag', action='store_true', help='Use residual connection')
  parser.add_argument('--residual_connection_method', type=str, default='sum', choices=("sum", "mean"), help='Residual connection method')
  parser.add_argument('--denoise_flag', action='store_true', help='Use denoise model')
  parser.add_argument('--in_denoiseRecons_lossFlag', action='store_true', help='Use builtIn denoise model')
  # parser.add_argument('--denoise_model', type=str, default=None, help='Denoise model')

  parser.add_argument('--vae_trainEpochs', type=int, default=100, help='Number of epochs to train vae')
  parser.add_argument('--vae_kernelNum', type=int, default=256, help='Number of kernels in the first layer of the VAE')
  parser.add_argument("--vae_zSize", type=int, default=2048, help="Size of the latent vector")
  parser.add_argument("--vae_lr", type=float, default=0.0001, help="VAE learning rate")
  parser.add_argument("--vae_weightDecay", type=float, default=1e-5, help="VAE Weight decay")
  parser.add_argument("--vae_lossFunc", default=False, help="Flag to use BCELoss for testing")  # not given loss_func, use original lossFunc 

  parser.add_argument('--GANs_trainEpochs', type=int, default=10, help='Number of epochs to train GANs')
  parser.add_argument('--GANs_latentDim', type=int, default=None, help='latent dim for GANs')
  parser.add_argument('--GANs_lr', type=float, default=0.0001, help='learning rate for GANs')
  parser.add_argument('--GANs_tensorboardComment', type=str, default='debug with GANs for resnet', help='tensorboard comment for GANs')
  # parser.add_argument('--vaeLatent_GANs', action='store_true', help='use vae latent dim for GANs')

  args = parser.parse_args()
  print(f"Script Arguments: {args}", flush=True)


  ############################
  # dataloader & model define (pretrain or not)
  ###########################
  classes_num = model_numClasses(args.dataset)
  if args.pretrained_flag:
    print('using pretrained resnet')
    # resnet = resnet18(pretrained=True)
    resnet = resnet18(weights='DEFAULT')
    if args.transfer_learning:
      args.addComment = 'transferLearning'
      for param in resnet.parameters():
        param.requires_grad = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, classes_num)   
  else:
    print('using non-pretrained resnet')
    resnet = resnet18(weights=None, num_classes=classes_num)

  resnet_boardComment, vae_boardComment = boardWriter_generator(args)
  print('RESNET board comment: ', resnet_boardComment)
  ############################
  ## dataset loader and define kernel_size
  ###########################
  if args.dataset in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CINIC10']:

    if args.norm:
      mean = (0.47889522, 0.47227842, 0.43047404)
      std = (0.24205776, 0.23828046, 0.25874835)
      transforms_smallSize = transforms.Compose([
      transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
      transforms.transforms.ToTensor(),
      transforms.Normalize(mean, std),])
    else:
      transforms_smallSize = transforms.Compose([
        # transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
        transforms.transforms.ToTensor(),
        ])
    
    dataset_loaders = create_dataloaders(transforms_smallSize, transforms_smallSize, args.batch_size, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)
    if args.dataset in ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN']:
      resnet.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
      resnet.maxpool = torch.nn.Identity()
    elif args.dataset in ['CINIC10']: 
      resnet.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
      resnet.maxpool = torch.nn.Identity()
  else:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms_largSize= transforms.Compose([
      transforms.Resize((256, 256),interpolation=InterpolationMode.BICUBIC),
      transforms.transforms.ToTensor(),
      transforms.Normalize(mean, std),])
    dataset_loaders = create_dataloaders(transforms_largSize, transforms_largSize, args.batch_size, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)
    resnet.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=3, bias=False)

  num_channel = dataset_loaders['train'].dataset[0][0].shape[0]
  image_size = dataset_loaders['train'].dataset[0][0].shape[1]

  # find the best lr, datasetloader, model, trainer_params, min_lr=1e-08, max_lr=1, training_epochs=100, lrFinder_method='fit')
  if args.customLr_flag and args.lr_scheduler_flag==False:
    suggested_lr = args.lr
    print("using given custom_lr: ", suggested_lr)
  elif args.lr_scheduler_flag:
    suggested_lr = args.lr
  else:
    suggested_lr = 0.0001

  
  ############################
  # denoise model
  ###########################
  if args.denoise_flag or args.augmentation_type == 'navie_denoiser':
    print('using denoise model, starts to train the denoise model')
    train_denoiseModel = DenoisingModel()
    denoiser_optimizer = torch.optim.Adam(train_denoiseModel.parameters(), lr=args.lr)
    denoiser_lrScheduler = lr_scheduler.ExponentialLR(denoiser_optimizer, gamma=0.9)
    for num in range(args.run_epochs):
      for batch_id, (img_tensor, label_tensor, id) in enumerate(dataset_loaders['train']):
        denoiser_optimizer.zero_grad()
        denoised_img = train_denoiseModel(img_tensor)
        denoiser_loss = F.mse_loss(denoised_img, img_tensor, size_average=False) / img_tensor.size(0)
        denoiser_loss.backward()
        denoiser_optimizer.step()
      if (num+1) in [20, 40, 60, 80, 100]:
        print(f'epoch: {num+1}, loss: {denoiser_loss.item()}')
      denoiser_lrScheduler.step()
    resnet_trainedDenoiser = train_denoiseModel
  else:
    resnet_trainedDenoiser = None


  ############################
  # Augmentation Part
  ###########################
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if args.augmentation_type == "simple":
    print('using ' + str(args.simpleAugmentation_name) + ' augmentation')
    simpleAugmentation_method = simpleAugmentation_selection(args.simpleAugmentation_name)
    augmentationType = args.augmentation_type
    augmentationTransforms = simpleAugmentation_method
    augmentationModel = None
    augmentationTrainer = None
  elif args.augmentation_type == "simple_crop" or (args.augmentation_type =='simple_centerCrop'):
    print('using ' + str(args.augmentation_type) + ' augmentation')
    augmentationType = args.augmentation_type
    augmentationTransforms = None
    augmentationModel = None
    augmentationTrainer = None
  #############################
  elif args.augmentation_type == "vae":
    print('using vae augmentation')
    vae_model = VAE(
        image_size=image_size,
        channel_num=num_channel,
        kernel_num=args.vae_kernelNum,
        z_size=args.vae_zSize,
        loss_func=args.vae_lossFunc,
    ).to(device)
    if args.reduce_dataset:
      vae_trainEpochs = 10
    else: 
      vae_trainEpochs = args.vae_trainEpochs
    train_model(vae_model, dataset_loaders,
            epochs=args.vae_trainEpochs,
            lr=args.lr,
            weight_decay=args.vae_weightDecay,
            tensorboard_comment = vae_boardComment,
            )
    augmentationType = args.augmentation_type
    augmentationTransforms = vae_augmentation
    augmentationModel = vae_model
    augmentationTrainer = None
  #############################
  elif args.augmentation_type == "navie_denoiser":
    print('using navie denoiser augmentation')
    augmentationType = args.augmentation_type
    augmentationTransforms = None
    augmentationModel = None
    augmentationTrainer = None
  #############################
  elif args.augmentation_type == 'builtIn_denoiser' or args.augmentation_type =='builtIn_vae':
    print(f'using {args.augmentation_type} augmentation')
    augmentationType = args.augmentation_type
    # the augmentation model would be 
    augmentationTransforms = None
    augmentationModel = None  
    augmentationTrainer = None
  #############################
  elif args.augmentation_type == "GANs":
    print('using GANs augmentation')
    if not isinstance(args.GANs_latentDim, int):
      print('using vae latent dim for GANs')
      gans_latent_dim = args.vae_zSize
      vae_for_gans = VAE(
      image_size=image_size,
      channel_num=num_channel,
      kernel_num=args.vae_kernelNum,
      z_size=args.vae_zSize,
      loss_func=args.vae_lossFunc,).to(device)
      train_model(vae_for_gans, dataset_loaders,
          epochs=args.GANs_trainEpochs,
          lr=args.lr,
          weight_decay=args.vae_weightDecay,
          tensorboard_comment = 'using vae latentDim for GANs',)
      augmentationTransforms = vae_gans_augmentation # passing the trained_vae model
    else:
      print('using given latent dim for GANs')
      gans_latent_dim = args.GANs_latentDim
      
    netD = Discriminator(in_channels=num_channel, image_size=image_size).to(device)
    netG = Generator(channel_num=num_channel, input_size=image_size, input_dim=gans_latent_dim).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)
    GANs_trainer = gans_trainer(netD=netD, netG=netG, dataloader=dataset_loaders, num_channel=num_channel, input_size=image_size, latent_dim=gans_latent_dim,
                           num_epochs=args.GANs_trainEpochs, batch_size=args.batch_size, lr=args.GANs_lr, criterion=torch.nn.BCELoss(),
                           tensorboard_comment=args.GANs_tensorboardComment)
    GANs_trainer.training_steps()
        # batch_images, _, _ = next(iter(dataset_loader['train']))
        # gans_vaeLatent_writer= SummaryWriter(comment="using vae latent for generating imgs with GANs")
        # batch_vaeLatent = vae.get_latent(batch_images.to(device))  #.view(2, -1)  # batch_vaeLatent.shape = (3, 128*4*4)
        # # new_size = (batch_vaeLatent.size(0), -1, 1, 1)
        # # batch_vaeLatent = batch_vaeLatent.view(new_size)
        # new_size = (batch_vaeLatent.size(0), batch_vaeLatent.size(1), 1, 1)
        # batch_vaeLatent = batch_vaeLatent.view(new_size)
        # result = trainer.get_imgs(batch_vaeLatent)  # input.shape = (batch_size, 128*4*4, 1, 1), output.shape = (batch_size, 3, 32, 32)
        # combine_imgs = torch.cat((batch_images[:8], result[:8]), 0)
        # gans_vaeLatent_writer.add_images("original vs vaeLatent_GANs_imgs", combine_imgs, dataformats="NCHW", global_step=0)
        # gans_vaeLatent_writer.close()

    augmentationType = args.augmentation_type
    augmentationTransforms = vae_gans_augmentation
    augmentationModel = vae_for_gans
    augmentationTrainer = GANs_trainer
  else:
    print('No augmentation')
    augmentationType = None
    augmentationTransforms = None
    augmentationModel = None
    augmentationTrainer = None

  if args.random_candidateSelection:
    print('randomly select candidates in this run')

  # if args.denoise_flag:
  #   print('using denoise model, starts to train the denoise model')
  #   train_denoiseModel = DenoisingModel()
  #   denoiser_optimizer = torch.optim.Adam(train_denoiseModel.parameters(), lr=0.0001)
  #   for num in range(args.run_epochs):
  #     for batch_id, (img_tensor, label_tensor, id) in enumerate(dataset_loaders['train']):
  #       denoiser_optimizer.zero_grad()
  #       denoised_img = train_denoiseModel(img_tensor)
#       denoiser_loss = F.mse_loss(denoised_img, img_tensor, size_average=False) / img_tensor.size(0)
  #       denoiser_loss.backward()
  #       denoiser_optimizer.step()
  #     if (num+1) in [20, 40, 60, 80, 100]:
  #       print(f'epoch: {num+1}, loss: {denoiser_loss.item()}')
  #   resnet_trainedDenoiser = train_denoiseModel
  # else:
  #   resnet_trainedDenoiser = None


  ############################
  if args.pretrained_flag:
    model_trainer = Resnet_trainer(dataloader=dataset_loaders, num_classes=classes_num, entropy_threshold=args.entropy_threshold, run_epochs=(args.run_epochs), start_epoch=args.candidate_start_epoch,
                                  model=resnet, loss_fn=torch.nn.CrossEntropyLoss(), individual_loss_fn=torch.nn.CrossEntropyLoss(reduction='none') ,optimizer= torch.optim.Adam, tensorboard_comment=resnet_boardComment,
                                  augmentation_type=augmentationType, augmentation_transforms=augmentationTransforms,
                                  augmentation_model=augmentationModel, model_transforms=augmentationTrainer,
                                  lr=suggested_lr, l2=args.l2, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,  # lr -- suggested_lr
                                  k_epoch_sampleSelection=args.k_epoch_sampleSelection,
                                  augmente_epochs_list=args.augmente_epochs_list,
                                  random_candidateSelection=args.random_candidateSelection, 
                                  residual_connection_flag=args.residualConnection_flag, residual_connection_method=args.residual_connection_method,
                                  denoise_flag=args.denoise_flag, denoise_model=resnet_trainedDenoiser,
                                  in_denoiseRecons_lossFlag = args.in_denoiseRecons_lossFlag,
                                  lr_scheduler_flag = args.lr_scheduler_flag,
                                  AugmentedDataset_func = args.AugmentedDataset_func,
                                  transfer_learning = args.transfer_learning,
                                )
  else:
    model_trainer = Resnet_trainer(dataloader=dataset_loaders, num_classes=classes_num, entropy_threshold=args.entropy_threshold, run_epochs=args.run_epochs, start_epoch=args.candidate_start_epoch,
                                    model=resnet, loss_fn=torch.nn.CrossEntropyLoss(), individual_loss_fn=torch.nn.CrossEntropyLoss(reduction='none') ,optimizer= torch.optim.Adam, tensorboard_comment=resnet_boardComment,
                                    augmentation_type=augmentationType, augmentation_transforms=augmentationTransforms,
                                    augmentation_model=augmentationModel, model_transforms=augmentationTrainer,
                                    lr=suggested_lr, l2=args.l2, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,  # lr -- suggested_lr
                                    k_epoch_sampleSelection=args.k_epoch_sampleSelection,
                                    augmente_epochs_list=args.augmente_epochs_list,
                                    random_candidateSelection=args.random_candidateSelection, 
                                    residual_connection_flag=args.residualConnection_flag, residual_connection_method=args.residual_connection_method,
                                    denoise_flag=args.denoise_flag, denoise_model=resnet_trainedDenoiser,
                                    in_denoiseRecons_lossFlag = args.in_denoiseRecons_lossFlag,
                                    lr_scheduler_flag = args.lr_scheduler_flag,
                                    AugmentedDataset_func = args.AugmentedDataset_func,
                                    transfer_learning = args.transfer_learning,
                                  )
  
  model_trainer.train()