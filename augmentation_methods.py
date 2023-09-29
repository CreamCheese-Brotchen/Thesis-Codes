import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
# from robust_attacks import CarliniL2
import pytorch_lightning as pl
# from pl_bolts.models.autoencoders import VAE

# autoencoder libraries
import urllib.parse
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import nn
from torch.nn import functional as F  # noqa: N812

# from pl_bolts import _HTTPS_AWS_HUB
# import pl_bolts.models.autoencoders.components
# from pl_bolts.models.autoencoders import VAE

from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)

from VAE_model import VAE



#################################################################################################################
class AugmentedDataset(Dataset):
    def __init__(self, dataset, target_idx_list, augmentation_transforms, 
                 augmentation_type, model=None, model_transforms=None,
                 tensorboard_epoch=None, tf_writer=None):
        self.dataset = dataset
        self.target_idx_list = target_idx_list
        self.augmentation_transforms = augmentation_transforms
        self.augmentation_type = augmentation_type
        self.model = model
        self.model_transforms = model_transforms
        self.tensorboard_epoch = tensorboard_epoch
        self.tf_writer = tf_writer

    def __getitem__(self, index):
        data, label, idx = self.dataset[index]

        # Apply data augmentation based on the index being in the target index list
        if idx in self.target_idx_list:
          if self.augmentation_type == 'vae':
            original_data = data
            data = self.model.get_singleImg(data).squeeze(0).detach().cpu()
            # data  = self.augmentation_transforms(data, self.model, self.model_transforms)  # apply_augmentation
            if self.tensorboard_epoch:   # store one pair of original and augmented images per epoch
              if idx in self.target_idx_list[-1]:
                combined_image = torch.cat((original_data, data), dim=2)  # Concatenate images side by side
                self.tf_writer.add_image('original & vae augmented imgs', combined_image, self.tensorboard_epoch)
          if self.augmentation_type == 'simple':
            original_data = data
            data  = self.augmentation_transforms(data)
            if self.tensorboard_epoch:   # store one pair of original and augmented images per epoch
              if idx in self.target_idx_list[-1]:
                combined_image = torch.cat((original_data, data), dim=2)  # Concatenate images side by side
                writer_comment = 'original & ' + str(self.augmentation_type) + ' augmented imgs'
                self.tf_writer.add_image(writer_comment, combined_image, self.tensorboard_epoch)
          if self.augmentation_type == 'GANs':
            original_data = data
            data  = self.augmentation_transforms(1).squeeze()  # 1: the num of imgs is 1, just one image; squeeze: remove the first dimension [1,3,32, 32] -> [3,32, 32]
            if self.tensorboard_epoch:   # store one pair of original and augmented images per epoch
              if idx in self.target_idx_list[-1]:
                combined_image = torch.cat((original_data, data), dim=2)  # Concatenate images side by side
                writer_comment = 'original & ' + str(self.augmentation_type) + ' augmented imgs'
                self.tf_writer.add_image(writer_comment, combined_image, self.tensorboard_epoch)

        return data, label, idx

    def __len__(self):
        return len(self.dataset)
    

#################################################################################################################
#### VAE Augmentation Methods
#################################################################################################################
def vae_augmentation(data, model, model_transforms=None):
    # data_pil = torch.utils.data.DataLoader([data], batch_size=1, shuffle=False)
    # augmented_data_pil = model_transforms.predict(model=model, dataloaders=data_pil) #model=model, trainer.predict()
    # augmented_data_resize = augmented_data_pil[0].squeeze()
    # augmented_data_tensor = augmented_data_resize
    model.eval()
    with torch.no_grad():
      augmented_data = model.get_singleImg(data)
    return augmented_data



#################################################################################################################
#### Simple Augmentation Methods
#################################################################################################################
def simpleAugmentation_selection(augmentation_name):
  if augmentation_name == "random_color":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(), transforms.ToTensor()])
  elif augmentation_name == "center_crop":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(150), transforms.Resize(256), transforms.ToTensor()])
  elif augmentation_name == "gaussian_blur":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), transforms.ToTensor()])
  elif augmentation_name == "elastic_transform":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.ElasticTransform(alpha=250.0), transforms.ToTensor()])
  elif augmentation_name == "random_perspective":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.RandomPerspective(), transforms.ToTensor()])
  elif augmentation_name == "random_resized crop":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.RandomResizedCrop(size=150), transforms.Resize(256), transforms.ToTensor()])
  elif augmentation_name == "random_invert":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.RandomInvert(p=0.9), transforms.ToTensor()])  
  elif augmentation_name == "random_posterize":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.RandomPosterize(bits=2), transforms.ToTensor()])
  elif augmentation_name == "rand_augment":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.RandAugment(), transforms.ToTensor()])
  elif augmentation_name == "augmix":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.AugMix(), transforms.ToTensor()])
  elif augmentation_name is None:
    augmentation_method = transforms.Compose([transforms.ToPILImage(), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
  ])
  else: 
    augmentation_method = transforms.Compose([transforms.ToPILImage(), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
  ])
  
  return augmentation_method






