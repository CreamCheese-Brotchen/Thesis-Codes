import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import torch
from torchvision import datasets
import torch.utils.data as data_utils
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image




class IndexDataset(Dataset):
    def __init__(self, Dataset):
        self.Dataset = Dataset
        
    def __getitem__(self, index):
        data, target = self.Dataset[index]
        
        return data, target, index

    def __len__(self):
        return len(self.Dataset)

class srgan_dataTransform(Dataset):
  def __init__(self, Dataset, hr_size, lr_size):
    self.Dataset = Dataset
    self.hr_size = hr_size
    self.lr_size = lr_size

    if self.hr_size is not None and self.lr_size is not None:
      assert self.hr_size[0] == 4 * self.lr_size[0]
      assert self.hr_size[1] == 4 * self.lr_size[1]

    self.hr_transforms = transforms.Compose([
            transforms.RandomCrop(hr_size),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
            # Low-res images are downsampled with bicubic kernel and scaled to [0, 1]
    self.lr_transforms = transforms.Compose([
        transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
        transforms.ToPILImage(),
        transforms.Resize(lr_size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])

    self.to_pil = transforms.ToPILImage()
    self.to_tensor = transforms.ToTensor()
    

  def __getitem__(self, index):
    data, target = self.Dataset[index]

    image = self.to_pil(data)

    hr = self.hr_transforms(image)
    lr = self.lr_transforms(hr)
    return hr, lr

  
  def __len__(self):
        return len(self.Dataset)

  @staticmethod
  def collate_fn(batch):
    hrs, lrs = [], []

    for hr, lr in batch:
        hrs.append(hr)
        lrs.append(lr)

    return torch.stack(hrs, dim=0), torch.stack(lrs, dim=0)
    


def create_dataloaders(transforms_train, transforms_test, batch_size, dataset_name, add_idx, reduce_dataset=False, srgan=False):
  if dataset_name == 'MNIST':
    data_train = datasets.MNIST(root='data', train=True, download=True, transform=transforms_train)         
    data_test = datasets.MNIST(root='data', train=False, download=True, transform=transforms_test)

  if dataset_name == 'FashionMNIST':
    data_train = datasets.FashionMNIST(root = 'data', train = True, download=True, transform = transforms_train)         
    data_test = datasets.FashionMNIST(root = 'data', train = False, download=True, transform = transforms_test)

  if dataset_name == 'CIFAR10': 
    data_train = datasets.CIFAR10(root = 'data', train = True, download=True, transform = transforms_train)         
    data_test = datasets.CIFAR10(root = 'data', train = False, download=True, transform = transforms_test)
    train_size = int(len(data_train) * 0.8) # 80% training data
    valid_size = len(data_train) - train_size
    data_train, data_valid = random_split(data_train, [train_size, valid_size])
  
  if dataset_name == 'SVHN':
    data_train = datasets.SVHN(root = 'data', split='train', download=True, transform = transforms_train)
    data_test = datasets.SVHN(root = 'data', split='test', download=True, transform = transforms_test)

  if dataset_name == 'Flowers102':
    data_train = datasets.Flowers102(root='./data', split='train', download=True, transform=transforms_train)
    data_test = datasets.Flowers102(root='./data', split='test', download=True, transform=transforms_train)

  if dataset_name == 'Food101':
    data_train = datasets.Food101(root='./data', split='train', download=True, transform=transforms_train)
    data_test = datasets.Food101(root='./data', split='test', download=True, transform=transforms_train)

  if dataset_name == 'CINIC10':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if str(device) == 'cpu':
      # cinic_directory = './data/CINIC-10'
      cinic_directory = 'D:/master_program/thesis/thesis-repo/data/CINIC-10'
    else:
      cinic_directory = './data/CINIC-10'
    data_train = datasets.ImageFolder(root=cinic_directory + '/train', transform=transforms_train)
    data_test = datasets.ImageFolder(root=cinic_directory + '/test', transform=transforms_test)
    data_valid = datasets.ImageFolder(root=cinic_directory + '/valid', transform=transforms_train)

  # debug
  if reduce_dataset:
    data_train = data_utils.Subset(data_train, torch.arange(32))
    data_valid = data_utils.Subset(data_valid, torch.arange(32))
    data_test = data_utils.Subset(data_test, torch.arange(32))  
 
  # give an unique id to every sample in the dataset to track the hard samples
  if add_idx:
      data_train = IndexDataset(data_train)
      data_valid = IndexDataset(data_valid)
      data_test = IndexDataset(data_test)

  if srgan:
    if add_idx:
      raise Exception("should not add idx for srgan_dataset")
    else:
      data_train = srgan_dataTransform(data_train, hr_size=[32,32], lr_size=[8,8])  


  dataset_loaders = {
          'train' : torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True),
          'valid' :torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=True), 
          'test'  : torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True),
          }

  return dataset_loaders


def model_numClasses(dataset_name):
  tenClasses = ['CIFAR10', 'SVHN', 'CINIC10','MNIST', 'FashionMNIST']
  if dataset_name in tenClasses:
    classes_num = 10
  if dataset_name == 'Flowers102':
    classes_num = 102
  if dataset_name == 'Food101':
    classes_num = 101
  return classes_num

def boardWriter_generator(args):

  resnet_comment = []
  
  if args.addComment:
    resnet_comment.append(args.addComment)

  if args.reduce_dataset:
    resnet_comment.append(' Debug')
  if args.pretrained_flag:
    resnet_comment.append('Pretrained')
  resnet_comment.append(args.dataset)


  ##############################
  ## augmentation 
  ##############################
  if args.augmentation_type == 'simple':
    resnet_comment.append(f"{ args.simpleAugmentation_name}")
  elif args.augmentation_type == 'simple_crop' or (args.augmentation_type == 'simple_centerCrop'):
    resnet_comment.append(f"{ args.augmentation_type}")
  elif args.augmentation_type == 'builtIn_denoiser':
    if args.in_denoiseRecons_lossFlag:
      resnet_comment.append(f"inDenoiser_totalLoss")
    else:
      resnet_comment.append(f" inDenoiser")
  elif args.augmentation_type == 'vae':
    resnet_comment.append(args.augmentation_type)
    if args.residualConnection_flag:
      resnet_comment.append(
        f" vae_{args.residual_connection_method}Residual",
      )
    if args.denoise_flag:
      resnet_comment.append(" vae_denoise")
  elif args.augmentation_type == 'navie_denoiser':
    resnet_comment.append(f" navieDenoiser")
  elif args.augmentation_type == 'builtIn_vae':
    builtIn_vae_comment = " builtIn_vae_lamda" + str(args.inAug_lamda)
    resnet_comment.append(builtIn_vae_comment)
    
  else:
    resnet_comment.append(f" noAug")


  ##############################
  ## basic
  ##############################
  if args.random_candidateSelection:
    resnet_comment.append(f"randomCandidate")
  elif args.k_epoch_sampleSelection == 0:
    resnet_comment.append(f"currentEpo_candidate")
  else:
    resnet_comment.append(f"{ args.k_epoch_sampleSelection}Epo_candidate")
  
  if args.norm:
    resnet_comment.append(f"norm")
  if args.lr_scheduler_flag:
    resnet_comment.append(f"lrScheduler")
  if args.transfer_learning:
    resnet_comment.append(f"transfer")

  resnet_comment.extend([
    f"{ args.entropy_threshold}ent",
    f"{ args.run_epochs}epo",
    f"{ args.candidate_start_epoch}se",
    f"lr_{ args.lr}",
    f"l2_{ args.l2}",
    f"{ args.batch_size}bs",
    # f"augDataloader_method{ args.AugmentedDataset_func}",
    ])
  if not args.augmentation_type:
    pass 
  else:
    resnet_comment.append(f"augMethod{args.AugmentedDataset_func}")

  vae_comment = []
  if args.augmentation_type == 'vae':
    vae_comment.extend([
      f"{args.dataset}",
      f"{args.augmentation_type}",
      f" Z_{args.vae_zSize}",
      f" K_{args.vae_kernelNum}"
      f" lr_{args.vae_lr}",
      f" l2_{args.vae_weightDecay}",
      f" {args.vae_trainEpochs}epo",
    ])

  
  return ' '.join(resnet_comment), ' '.join(vae_comment)
  # if args.
    
    # basic.extend([
    #   f"{args['augmentation_type']}Aug",
    #   f"vaeZ_{args['vae_zSize']}",
    #   f"vaeLr_{args['vae_lr']}",
    # ])


