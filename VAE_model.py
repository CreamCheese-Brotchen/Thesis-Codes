import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import nn
from torch.nn import functional as F  
import matplotlib.pyplot as plt
import torchvision
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
import argparse
from dataset_loader import IndexDataset, create_dataloaders, model_numClasses
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter



class VAE(LightningModule):
    def __init__(
        self,
        input_height: int,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ) -> None:
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            "resnet18": {
                "enc": resnet18_encoder,
                "dec": resnet18_decoder,
            },
            "resnet50": {
                "enc": resnet50_encoder,
                "dec": resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]["dec"](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)


    def forward(self, x):
        # x = x[0]   # added this line because of the size of the dataloader problem
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):  # should be '_get_reconstruction_loss' ? 
        x, y, id  = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    
class MyVAE_test(VAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    #     # Add any additional customization here
    def training_step(self, batch, batch_idx):
        x, _ = batch
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = reconstruction_loss + self.kl_weight * kl_divergence
        
        self.log('train_loss', loss)
        self.log('reconstruction_loss', reconstruction_loss)
        self.log('kl_divergence', kl_divergence)
        
        return loss



class MyVAE(LightningModule):
    def __init__(
        self,
        input_height: int,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ) -> None:

        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            "resnet18": {
                "enc": resnet18_encoder,
                "dec": resnet18_decoder,
            },
            "resnet50": {
                "enc": resnet50_encoder,
                "dec": resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]["dec"](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)


    def forward(self, x):
        # x = x[0]   # added this line because of the size of the dataloader problem
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.reparameterize(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.reparameterize(mu, log_var)
        return z, self.decoder(z), p, q

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):  # should be '_get_reconstruction_loss' ? 
        x, y, id  = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resnet Training script')

    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=("MNIST", "CIFAR10", "FashionMNIST", "SVHN", "Flowers102", "Food101"), help='Dataset name')
    parser.add_argument('--run_epochs', type=int, default=5, help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (default: 64)')
    parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')
    parser.add_argument('--tensorboard_comment', type=str, default='test_run', help='Comment to append to tensorboard logs')
    args = parser.parse_args()
    print(f"Script Arguments: {args}", flush=True)



    writer = SummaryWriter(comment=args.tensorboard_comment)

    transforms_smallSize = transforms.Compose([
        transforms.transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataloader = create_dataloaders(transforms_smallSize, transforms_smallSize, 64, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = pl.Trainer(max_epochs=args.run_epochs, accelerator=str(device), auto_lr_find=True)  # Customize Trainer options as needed
    # train_dataloader = create_dataloaders(transforms_smallSize, transforms_smallSize, 16, "CIFAR10", add_idx=True, reduce_dataset=True)
    # trainer = pl.Trainer(max_epochs=30, auto_lr_find=True)

    my_vae = MyVAE(input_height=32, latent_dim=256, learning_rate=1e-3, kl_weight=0.1)
    trainer.tune(my_vae, train_dataloader['train'])
    trainer.fit(my_vae, train_dataloader['train'])

# Assuming you have trained your MyVAE model and loaded the checkpoint

# Set the model to evaluation mode
    my_vae.eval()

# Generate a batch of input data
    input_data, _, _ = next(iter(train_dataloader["train"]))
    writer.add_images('original_images', input_data, 0)
    # Pass the input data through the model to get reconstructed images
    with torch.no_grad():
        # mu, log_var = my_vae.encoder(input_data)
        # z = my_vae.reparameterize(mu, log_var)
        reconstructed_data = my_vae.forward(input_data)
        writer.add_images('vae reconstructed_images', reconstructed_data, 0)


# Now you can visualize the original and reconstructed images
# original_images = torchvision.utils.make_grid(input_data, nrow=8, normalize=True)
# reconstructed_images = torchvision.utils.make_grid(reconstructed_data, nrow=8, normalize=True)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Original Images')
# plt.imshow(np.transpose(original_images, (1, 2, 0)))

# plt.subplot(1, 2, 2)
# plt.title('Reconstructed Images')
# plt.imshow(np.transpose(reconstructed_images, (1, 2, 0)))

# plt.tight_layout()
# plt.show()
