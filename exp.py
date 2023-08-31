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
import seaborn as sns
import matplotlib
from more_itertools import flatten
import itertools
from collections import Counter
import copy
from pytorch_lightning import LightningModule, Trainer
# from memory_profiler import profile
# import sys 

from dataset_loader import IndexDataset, create_dataloaders, model_numClasses
from augmentation_methods import simpleAugmentation_selection, AugmentedDataset, vae_augmentation
from VAE_model import VAE
from resnet_model import Resnet_trainer


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Resnet Training script')

  parser.add_argument('--dataset', type=str, default='CIFAR10', choices=("MNIST", "CIFAR10", "FashionMNIST", "SVHN", "Flowers102", "Food101"), help='Dataset name')
  parser.add_argument('--entropy_threshold', type=float, default=0.5, help='Entropy threshold')
  parser.add_argument('--run_epochs', type=int, default=5, help='Number of epochs to run')
  parser.add_argument('--candidate_start_epoch', type=int, default=0, help='Epoch to start selecting candidates. Candidate calculation begind after the mentioned epoch')
  parser.add_argument('--tensorboard_comment', type=str, default='test_run', help='Comment to append to tensorboard logs')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--l2', type=float, default=1e-4, help='L2 regularization')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
  parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')
  parser.add_argument('--augmentation_type', type=str, default=None, choices=("vae", "simple"), help='Augmentation type')
  parser.add_argument('--simpleAugmentaion_name', type=str, default=None, choices=("random_color", "center_crop", "gaussian_blur", 
                                                                                   "elastic_transform", "random_perspective", "random_resized_crop", 
                                                                                   "random_invert", "random_posterize", "rand_augment", "augmix"), help='Simple Augmentation name')
  parser.add_argument('--accumulation_steps', type=int, default=None, help='Number of accumulation steps')
  parser.add_argument('--vae_accumulationSteps', type=int, default=4, help='Accumulation steps for VAE training')
  parser.add_argument('--k_epoch_sampleSelection', type=int, default=None, help='Number of epochs to select the common candidates')
  parser.add_argument('--augmente_epochs_list', type=list, default=None, help='Number of epochs to train VAE')
  args = parser.parse_args()
  print(f"Script Arguments: {args}", flush=True)


  resnet = resnet18(weights=None)
  classes_num = model_numClasses(args.dataset)
  if args.dataset in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10']:
    mean = (0.5,)
    std = (0.5, 0.5, 0.5) 
    transforms_smallSize = transforms.Compose([
      # transforms.Resize((32, 32)),
      transforms.transforms.ToTensor(),
      # transforms.Normalize(mean, mean),
      ])
    dataset_loaders = create_dataloaders(transforms_smallSize, transforms_smallSize, args.batch_size, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)
    resnet.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
  else:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms_largSize= transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.transforms.ToTensor(),
    transforms.Normalize(mean, std),])
    dataset_loaders = create_dataloaders(transforms_largSize, transforms_largSize, args.batch_size, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)
    resnet.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=3, bias=False)
  
  num_ftrs = resnet.fc.in_features
  resnet.fc = torch.nn.Linear(num_ftrs, classes_num)  
  print(f"Number of classes: {classes_num}", flush=True)

  

  if args.augmentation_type == "simple":
    print('using simple augmentation')
    simpleAugmentation_method = simpleAugmentation_selection(args.simpleAugmentaion_name)
    augmentationType = 'simple'
    augmentationTransforms = simpleAugmentation_method
    augmentationModel = None
    augmentationTrainer = None
  elif args.augmentation_type == "vae":
    # firstly, trian the datset
    print('using vae augmentation')
    input_height = 32
    vae_model = VAE(input_height=input_height)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.reduce_dataset:
      vae_trainEpochs = 10
    else: 
      vae_trainEpochs = 100
    vae_trainer = Trainer(max_epochs=vae_trainEpochs, accumulate_grad_batches=args.vae_accumulationSteps, accelerator="auto", strategy="auto", devices="auto", enable_progress_bar=False)
    vae_trainer.tune(vae_model, dataset_loaders['train'])
    vae_trainer.fit(vae_model, dataset_loaders['train'])
    # passing the vae trainer to the model_trainer
    augmentationType = 'vae'
    augmentationTransforms = vae_augmentation
    augmentationModel = vae_model
    augmentationTrainer = vae_trainer
  elif args.augmentation_type == None:
    print('No augmentation')
    augmentationType = None
    augmentationTransforms = None
    augmentationModel = None
    augmentationTrainer = None

  model_trainer = Resnet_trainer(dataloader=dataset_loaders, num_classes=classes_num, entropy_threshold=args.entropy_threshold, run_epochs=args.run_epochs, start_epoch=args.candidate_start_epoch,
                                  model=resnet, loss_fn=torch.nn.CrossEntropyLoss(), individual_loss_fn=torch.nn.CrossEntropyLoss(reduction='none') ,optimizer= torch.optim.Adam, tensorboard_comment=args.tensorboard_comment,
                                  augmentation_type=augmentationType, augmentation_transforms=augmentationTransforms,
                                  augmentation_model=augmentationModel, model_transforms=augmentationTrainer,
                                  lr=args.lr, l2=args.l2, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps
                                  )
  model_trainer.train()