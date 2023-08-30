import numpy as np
import matplotlib
import pandas as pd
import torch
import torch.utils.data as data_utils
from torchvision import transforms
import torchmetrics
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from VAE_model import VAE
from dataset_loader import IndexDataset, create_dataloaders, model_numClasses
import argparse
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Resnet Training script')

  parser.add_argument('--dataset', type=str, default='CIFAR10', choices=("MNIST", "CIFAR10", "FashionMNIST", "SVHN", "Flowers102", "Food101"), help='Dataset name')
  parser.add_argument('--vae_runEpochs', type=int, default=5, help='Number of epochs to run')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
  parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')
  parser.add_argument('--tensorboard_comment', type=str, default='test_run', help='Comment to append to tensorboard logs')
  args = parser.parse_args()
  print(f"Script Arguments: {args}", flush=True)

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transforms_smallSize = transforms.Compose([
  transforms.transforms.ToTensor(),
])

dataset_loaders = create_dataloaders(transforms_smallSize, transforms_smallSize, args.batch_size, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)

input_height = 32
vae = VAE(input_height=input_height)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainer = Trainer(max_epochs=args.vae_runEpochs , accelerator=str(device), auto_lr_find=True)
trainer.tune(vae, dataset_loaders['train'])
trainer.fit(vae, dataset_loaders['train'])

writer = SummaryWriter(comment=args.tensorboard_comment)
for batch_id, (img_tensor, label_tensor, id) in enumerate(dataset_loaders['test']):
  first_img = img_tensor[0]
  original_img = torch.utils.data.DataLoader(img_tensor, batch_size=1, shuffle=False)
  vae_out= trainer.predict(model=vae, dataloaders=original_img)
  first_vaeImg = vae_out[0].squeeze()  # first the vae_img
#   augmented_data_tensor = first_vaeImg
  combined_image = torch.cat((first_img, first_vaeImg), dim=2)
  writer.add_image('original vs VAE img', combined_image, batch_id)


  

