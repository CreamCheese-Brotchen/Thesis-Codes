import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
# autoencoer libraries
import urllib.parse
from argparse import ArgumentParser
import torch
from torch.nn import functional as F



#################################################################################################################
class AugmentedDataset(Dataset):
    def __init__(self, dataset, target_idx_list, augmentation_transforms, 
                 augmentation_type, model=None, model_transforms=None,
                 tensorboard_epoch=None, tf_writer=None,
                 residual_connection_flag=False, residual_connection_method=None,
                 denoise_flag=False, denoise_model=None,
                 builtIn_denoise_model=None,
                 in_denoiseRecons_lossFlag=False):
        self.dataset = dataset
        self.target_idx_list = target_idx_list
        self.augmentation_transforms = augmentation_transforms
        self.augmentation_type = augmentation_type
        self.model = model
        self.model_transforms = model_transforms
        self.tensorboard_epoch = tensorboard_epoch
        self.tf_writer = tf_writer

        self.residual_connection_flag = residual_connection_flag
        self.residual_connection_method = residual_connection_method
        self.denoise_flag = denoise_flag
        self.denoise_model = denoise_model

        # built-in denoiser
        self.builtIn_denoise_model = builtIn_denoise_model
        self.in_denoiseRecons_lossFlag = in_denoiseRecons_lossFlag

    def __getitem__(self, index):
        data, label, idx = self.dataset[index]
      
        # Apply data augmentation based on the index being in the target index list
        if idx in self.target_idx_list:
          if self.augmentation_type == 'vae':
            original_data = data
            data = self.model.get_singleImg(data.to(self.model.device)).squeeze(0).to(original_data.device)  # [3, 32, 32], get the augmented img from the model
            # data  = self.augmentation_transforms(data, self.model, self.model_transforms)  # apply_augmentation

            tf_imgComment = 'Resnet_Aug/Orig & vaeAug'

            if self.residual_connection_flag:
              if self.residual_connection_method[0] == 'sum':
                data = data + original_data
                data = torch.clip(data, 0, 1)
              elif self.residual_connection_method[0] == 'mean':
                data = torch.add(data, original_data)/2
                data = torch.clip(data, 0, 1)
              tf_imgComment += ' & resCon_' + str(self.residual_connection_method[0])

            if self.denoise_flag:
              # vae-> denoiser = denoiser_data; original_data + denoiser_data = augmented_data
              denoiser_data = self.denoise_model(data.unsqueeze(0))
              data = original_data + denoiser_data.squeeze(0).detach()
              tf_imgComment += ' & denoisImg'

            if self.tensorboard_epoch:   # store one pair of original and augmented images per epoch
              if idx in self.target_idx_list[-1]:
                combined_image = torch.cat((original_data, data.detach()), dim=2)  # Concatenate images side by side
                self.tf_writer.add_image(str(tf_imgComment), combined_image, self.tensorboard_epoch)

          if self.augmentation_type == 'simple':
            original_data = data
            data  = self.augmentation_transforms(data)

            if self.tensorboard_epoch:   # store one pair of original and augmented images per epoch
              if idx in self.target_idx_list[-1]:
                combined_image = torch.cat((original_data, data), dim=2)  # Concatenate images side by side
                tf_imgComment = 'Resnet_Aug/Orig & ' + str(self.augmentation_type) + 'Aug'
                self.tf_writer.add_image(tf_imgComment, combined_image, self.tensorboard_epoch)

          if self.augmentation_type == 'navie_denoiser':
            original_data = data
            data  = self.denoise_model(data.unsqueeze(0))
            data = original_data + data.squeeze(0).detach()
            if self.tensorboard_epoch:
              if idx in self.target_idx_list[-1]:
                combined_image = torch.cat((original_data, data), dim=2)  # Concatenate images side by side
                self.tf_writer.add_image('Resnet_Aug/Orig & navieDenoise', combined_image, self.tensorboard_epoch)

          if self.augmentation_type == 'builtIn_denoiser':
            original_data = data
            data  = self.builtIn_denoise_model.get_singleImg(data.unsqueeze(0)).squeeze(0)
            if self.tensorboard_epoch:
              if idx in self.target_idx_list[-1]:
                combined_image = torch.cat((original_data, data.detach()), dim=2)
                comment = 'Resnet_Orig/Aug & builtIn_denoiser img'
                if self.in_denoiseRecons_lossFlag:
                   comment+= '(totaLoss)'
                self.tf_writer.add_image(comment, combined_image, self.tensorboard_epoch)

          if self.augmentation_type == 'GANs':
            original_data = data  
            data = self.augmentation_transforms(data, latent_model=self.model, augmentation_model=self.model_transforms).to(original_data.device)  # model: vae_model, model_transforms: GANs_trainer
            # data  = self.augmentation_transforms(1).squeeze()  # 1: the num of imgs is 1, just one image; squeeze: remove the first dimension [1,3,32, 32] -> [3,32, 32]
            if self.tensorboard_epoch:   # store one pair of original and augmented images per epoch
              if idx in self.target_idx_list[-1]:
                combined_image = torch.cat((original_data, data), dim=2)  # Concatenate images side by side
                writer_comment = 'Resnet_aug/original & ' + str(self.augmentation_type) + ' augmented imgs'
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
#### VAE Augmentation Methods
#################################################################################################################
def vae_gans_augmentation(data, latent_model, augmentation_model):
    latent_model.eval()
    with torch.no_grad():
      vae_latent = latent_model.get_latent(data.unsqueeze(0).to(latent_model.device))
    new_size = (vae_latent.size(0), vae_latent.size(1), 1, 1)
    vae_latent = vae_latent.view(new_size)
    augmented_data = augmentation_model.get_imgs(vae_latent)
    return augmented_data.squeeze()  # [3, 32, 32]


#################################################################################################################
#### Simple Augmentation Methods
#################################################################################################################
def simpleAugmentation_selection(augmentation_name):
  if augmentation_name == "random_color":
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(brightness=.5, hue=.3), transforms.ToTensor()])
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


#################################################################################################################
#### residual connections
#################################################################################################################
def residual_connect(x_constructed, x_original, method='sum'):
  if method == 'sum':
    return x_constructed + x_original
  elif method == 'mean':
    return (x_constructed + x_original) / 2
  

#################################################################################################################
#### Augmentation Methods
#################################################################################################################
def non_local_op(l, softmax=True, embed=False):

    batch_size = list(l.shape)[0]
    n_in, H, W = list(l.shape)[1:]

    if embed:
        theta_layer = torch.nn.Conv2d(in_channels=n_in, out_channels=n_in, kernel_size=1, padding='same')
        phi_layer = torch.nn.Conv2d(in_channels=n_in, out_channels=n_in, kernel_size=1, padding='same')

        theta = theta_layer(l)
        phi = phi_layer(l)
        g = l
    else:
        theta, phi, g = l, l, l
        phi = phi.view(batch_size, n_in, -1).permute(0, 2, 1)
        theta = theta.view(batch_size, n_in, -1)
        g = g.view(batch_size, n_in, -1).permute(0, 2, 1)

    if n_in > H * W or softmax:

        f = torch.einsum('bij,bjk->bik', phi, theta) 

        if softmax:
            f = f / torch.sqrt(torch.tensor(theta.shape[1]))
            f = F.softmax(f, dim=2)
        f = torch.einsum('bik,bkj->bij', f, g)
        f = f.permute(0, 2, 1)
    else:
        f = torch.einsum('nihw,njhw->nij', [phi, g])
        f = torch.einsum('nij,nihw->njhw', [f, theta])
    if not softmax:
        f = f / torch.tensor(H * W, dtype=f.dtype)
    return f.view(l.size())


class DenoisingModel(torch.nn.Module):
    def __init__(self):
        super(DenoisingModel, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding='same')
        
    def forward(self, input):
        f = non_local_op(input, softmax=True, embed=False)
        f1 = self.conv(f)
        output = input + f1
        return output

    def get_singleImg(self, x):
        with torch.no_grad():
           x_recons = self.forward(x)
        return x_recons

