import torch
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from dataset_loader import IndexDataset, create_dataloaders, model_numClasses
import torch.nn as nn




class LightningTransformer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()        # self.model = model_selection(model_name, num_classes)
        self.model = model

    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        imgs, labels, id = batch
        preds = self.model(imgs)
        loss = nn.CrossEntropyLoss()(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tun_model = resnet18(pretrained=False, num_classes=10).to(device)
# transformed_model = LightningTransformer(model=tun_model)
# trainer = pl.Trainer(max_epochs=10)
# lr_finder  = trainer.tuner.lr_find(model=transformed_model, train_dataloaders=dataset_loaders['train'], min_lr=1e-08, max_lr=1, method='fit')
# # param = trainer.tuner.scale_batch_size(model=model, train_dataloaders=dataset_loaders['train'], max_trials=10, mode='power')
# print(lr_finder.suggestion())

# fig = lr_finder.plot(); fig.show()
# suggested_lr = lr_finder.suggestion()
class lrSearch():
    def __init__(self, datasetloader, model, trainer_params, min_lr=1e-08, max_lr=1, training_epochs=100, lrFinder_method='fit'):
        self.datasetloader = datasetloader
        self.model = model
        self.trainer_params = trainer_params
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lrFinder_method = lrFinder_method
        self.training_epochs = training_epochs

    def search(self):
        transformed_model = LightningTransformer(model=self.model)    
        trainer = pl.Trainer(**self.trainer_params)
        lr_finder  = trainer.tuner.lr_find(model=transformed_model, train_dataloaders=self.datasetloader, min_lr=self.min_lr, max_lr=self.max_lr, num_training=self.training_epochs, method=self.lrFinder_method)
        return lr_finder.suggestion()

     

