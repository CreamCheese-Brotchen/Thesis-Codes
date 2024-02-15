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
from collections import Counter
import matplotlib
# import tensorflow as tf
from more_itertools import flatten
from torch.optim import lr_scheduler
import itertools
from collections import Counter
import copy
from pytorch_lightning import LightningModule, Trainer
import pandas as pd
from augmentation_folder.dataset_loader import IndexDataset, create_dataloaders
from augmentation_folder.augmentation_methods import simpleAugmentation_selection, AugmentedDataset, AugmentedDataset2, vae_augmentation, DenoisingModel, create_augmented_dataloader
from VAE_folder.VAE_model import VAE
import random
# from memory_profiler import profile
# import sys 

# renetIn_denoiser = DenoisingModel()

class Resnet_trainer():
  def __init__(self, dataloader, num_classes, entropy_threshold, run_epochs, start_epoch, model,
              loss_fn, individual_loss_fn, optimizer, tensorboard_comment,
              augmentation_type=None, augmentation_transforms=None,
              augmentation_model=None, model_transforms=None,
              lr=0.0001, l2=0, batch_size=64, accumulation_steps=None,
              k_epoch_sampleSelection=3,
              random_candidateSelection=False,
              augmente_epochs_list=None,
              residual_connection_flag=False, residual_connection_method=None,   # resConnect/Denoise for vae
              denoise_flag=False, denoise_model=None,                            # resConnect/Denoise for vae
              in_denoiseRecons_lossFlag=False,
              lr_scheduler_flag = False,
              AugmentedDataset_func=1,
              transfer_learning = False, inAug_lamda = 0.7,
              ):                                     # built-in denoisers
    self.dataloader = dataloader
    self.entropy_threshold = entropy_threshold
    self.run_epochs = run_epochs
    self.start_epoch = start_epoch
    self.model = model
    self.lr = lr
    self.l2 = l2
    self.loss_fn = loss_fn      # torch.nn.CrossEntropyLoss()
    self.individual_loss_fn = individual_loss_fn
    self.transfer_learning = transfer_learning
    if self.transfer_learning:
      self.optimizer = optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.l2) #optimizer(self.model.fc.parameters(), lr=self.lr, weight_decay=self.l2)
    else:
      self.optimizer = optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.l2)
    # tested for transfer-learning, torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)
    self.tensorboard_comment = tensorboard_comment
    self.lr_scheduler_flag = lr_scheduler_flag
    if self.lr_scheduler_flag:
      self.lr_Scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    self.augmentation_type = augmentation_type
    self.augmentation_transforms = augmentation_transforms
    self.augmentation_model = augmentation_model
    self.model_transforms = model_transforms
    # replace the original as new dataset: 2 or temporily dataset1
    self.AugmentedDataset_func = AugmentedDataset_func
    # resConnect/Denoise for vae
    self.residual_connection_flag=residual_connection_flag
    self.residual_connection_method=residual_connection_method
    self.denoise_flag = denoise_flag
    self.denoise_model = denoise_model
    # builtin_denoise
    self.builtin_denoise_model = DenoisingModel()
    self.denoiser_optimizer = torch.optim.Adam(self.builtin_denoise_model.parameters(), lr=self.lr)
    self.indenoiser_lrSceduler = lr_scheduler.ExponentialLR(self.denoiser_optimizer, gamma=0.9)
    self.denoiser_loss = torch.nn.CrossEntropyLoss()
    self.in_denoiseRecons_lossFlag = in_denoiseRecons_lossFlag
    # basic params
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.accumulation_steps  = accumulation_steps # gradient accumulation steps
    self.inAug_lamda = inAug_lamda
    self.augmente_epochs_list = augmente_epochs_list  # number of epochs for augmentation, list [20, 30, ...., 90]
    self.k_epoch_sampleSelection = k_epoch_sampleSelection
    self.random_candidateSelection = random_candidateSelection
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if self.augmentation_type == 'builtIn_vae':
      imgs, _, _ = next(iter(self.dataloader['train']))
      image_size = imgs[0].size()
      print(f"image_size {image_size[1]}, num_channel {image_size[0]}"	)
      self.reset_vae = VAE(image_size=image_size[1], channel_num=image_size[0], kernel_num=256, z_size=1024, loss_func=None).to(self.device)
      self.resnet_vae_optimizer = torch.optim.Adam(self.reset_vae.parameters(), lr=self.lr)
      self.inVae_lrScheduler = lr_scheduler.ExponentialLR(self.resnet_vae_optimizer, gamma=0.9)

  def train_builtIn_denoiser(self, curentIter, currentIter_resnetLoss, img):
    # print(f"currentIter_loss {currentIter_loss}")
    with torch.no_grad():
      denoiser_output = self.builtin_denoise_model(img)
      denoiser_loss = self.denoiser_loss(denoiser_output, img)

    self.denoiser_optimizer.zero_grad()
    totalLoss = currentIter_resnetLoss + denoiser_loss
    totalLoss.backward()
    self.denoiser_optimizer.step()
    if curentIter % 100 == 0:
      print(f"currentIter_loss {totalLoss}")
    return self.builtin_denoise_model

  def selection_candidates(self, current_allId_list, current_allEnt_list, current_allLoss_list, history_candidates_id,
                          #  history_entropy_candidates, history_num_candidates, history_meanLoss_candidates,
                           randomCandidate_selection=False):
    """Input current id/entropy/loss of all samples, output the selected candidates with entropy > threshold, and update the history of candidates
    Args:
        current_allId_list (list): current id list of all samples
        current_allEnt_list (list): cuurent entropy list of all samples
        current_allLoss_list (list): current loss list of all samples
        history_candidates_id (list): history of candidates id collected across  epochs
        history_entropy_candidates (list): history of candidates entropy collected across  epochs
        history_num_candidates (list): history of candidates number collected across  epochs
        history_meanLoss_candidates (list):  history of candidates mean loss collected across epochs

    Returns:
        _type_: update the history of candidates: id, entropy, number, mean loss, which are collected across epochs
    """
    idList_allSamples = list(flatten(current_allId_list))
    entropy_allSamples = list(flatten(current_allEnt_list))   # flattened, a tensor list of entropy of candidates over batchs
    loss_allSamples = list(flatten(current_allLoss_list))
    # class_allSamples = list(flatten(current_allClass_list))

    if randomCandidate_selection:
      combined_info = list(zip(idList_allSamples, entropy_allSamples, loss_allSamples))
      np.random.shuffle(combined_info)
      random_candidateNum = int(np.random.randint(0.15*len(idList_allSamples), 0.75*len(idList_allSamples)))  # num random_candidates
      select_Candidates = combined_info[:random_candidateNum]
      print(f"randomly select {random_candidateNum} candidates at this epoch")
    else:
      select_Candidates = [(idx,ent,loss) for (idx,ent,loss) in zip(idList_allSamples,entropy_allSamples,loss_allSamples) if ent >= self.entropy_threshold]
      # select_Candidates = [(idx,ent,loss,class_id) for (idx,ent,loss,class_id) in zip(idList_allSamples, entropy_allSamples, loss_allSamples, class_allSamples) if ent >= self.entropy_threshold]
    # select_Candidates = [(idx,ent) for (idx,ent) in zip(idList_allSamples,entropy_allSamples) if ent >= self.entropy_threshold]
    if len(select_Candidates):
      candidates_id, candidates_entropy, candidates_loss = list(zip(*select_Candidates))
      history_candidates_id.append(candidates_id)        # to search the past k epochs for searching more stubborn candidates
      candidates_loss_cpu = [loss.cpu().detach().numpy() for loss in candidates_loss]
      currentEpoch_lossCandidate = candidates_loss_cpu  #np.mean(candidates_loss_cpu)
    else:
      candidates_id = []
      history_candidates_id.append([])
      currentEpoch_lossCandidate = np.nan
      candidates_entropy = []
      # candidates_class = []

    return history_candidates_id, currentEpoch_lossCandidate, candidates_id, candidates_entropy
    # return history_num_candidates, history_meanLoss_candidates, history_candidates_id, history_entropy_candidates, candidates_id


  def commonId_k_epochSelection(self, history_candidates_id, k):
    lists_dict = {}
    previousEpoch_selection = min(len(history_candidates_id), k)

    for i in range(previousEpoch_selection):
      list_name = f"list_{i}"
      lists_dict[list_name] = set(history_candidates_id[-(i+1)])  # finding unique elements in the list

    common_id = None
    for j in range(len(lists_dict)):
      t1 = list(lists_dict.items())[j:]
      common_id = set.intersection(*dict(t1).values())
      # print(j, common_id)
      if common_id:
        break
    return common_id


  def train(self):

    # generate the augmente_epochs_list
    if (self.run_epochs >= 20) and (self.augmente_epochs_list is None):
              self.augmente_epochs_list =  np.arange(self.start_epoch, self.run_epochs, 10)  # every 10 epochs (20, 30, ..., 90), augment the dataset
              self.change_augmente_epochs_list = self.augmente_epochs_list + 1
    elif (self.run_epochs < 20) and (self.augmente_epochs_list is None):
            self.augmente_epochs_list =  np.arange(self.start_epoch, self.run_epochs, 2)  # debug mode, every 2 epochs, augment the dataset
            self.change_augmente_epochs_list = self.augmente_epochs_list + 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)
    self.builtin_denoise_model.to(device)

    # basic train/test loss/accuracy
    avg_loss_metric_train = torchmetrics.MeanMetric().to(device)
    accuracy_metric_train = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device)
    avg_loss_metric_test = torchmetrics.MeanMetric().to(device)
    accuracy_metric_test = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device)

    # built-in augmentation model
    denoiserLoss_metric = torchmetrics.MeanMetric().to(device)
    resnet_vae_metric = torchmetrics.MeanMetric().to(device)

    if self.AugmentedDataset_func == 1:
      # define the training dataset -- temporartily replace the data with augmented data
      augmented_dataset = AugmentedDataset(
          dataset = self.dataloader['train'].dataset,
          target_idx_list = [],
          augmentation_type = self.augmentation_type,
          augmentation_transforms = self.augmentation_transforms,
          model = self.augmentation_model,
          model_transforms = self.model_transforms,
          tensorboard_epoch = [],  # NEW
          tf_writer = [],
          )
      train_dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=self.batch_size, shuffle=True)
    elif self.AugmentedDataset_func == 2 or self.AugmentedDataset_func == 3:
      # competely replace the data with augmented data -- create new dataset
      train_dataloader = self.dataloader['train']
      # augmented_dataset = AugmentedDataset2(
      #     dataset = self.dataloader['train'].dataset,
      #     target_idx_list = [],
      #     augmentation_type = self.augmentation_type,
      #     augmentation_transforms = self.augmentation_transforms,
      #     model = self.augmentation_model,
      #     model_transforms = self.model_transforms,
      #     tensorboard_epoch = [],  # NEW
      #     tf_writer = [],
      # )
      # train_dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=self.batch_size, shuffle=True)

    # tensorboard
    writer = SummaryWriter(comment=self.tensorboard_comment)
    print(f"Starting Training Run. Using device: {device}", flush=True)

    history_candidates_id = list()
    valid_history_candidates_id = list()

    for epoch in range(self.run_epochs):

      id_list = list()                  # id of all samples at each epoch, cleaned when the new epoch starts
      entropy_list = list()             # entropy of all samples at each epoch, histogram, cleaned when the new epoch starts
      all_individualLoss_list = list()  # loss of all samples at each epoch
      # class_list = list()

      valid_id_list = list()                  # id of all samples at each epoch, cleaned when the new epoch starts
      valid_entropy_list = list()             # entropy of all samples at each epoch, histogram, cleaned when the new epoch starts
      valid_all_individualLoss_list = list()  # loss of all samples at each epoch

      test_entropy_list = list()             # entropy of all samples at each epoch, histogram, cleaned when the new epoch starts'
      test_id_list = list()                  # id of all samples at each epoch, cleaned when the new epoch starts
      test_all_individualLoss_list = list()


      self.model.train()
      if not self.augmentation_type:
        for batch_id, (img_tensor, label_tensor, id) in enumerate(self.dataloader['train']):
          self.model.train()
          img_tensor = Variable(img_tensor).to(device)
          label_tensor = Variable(label_tensor).to(device)
          output_logits = self.model(img_tensor)

          output_probs = torch.nn.Softmax(dim=1)(output_logits)
          loss_val = self.loss_fn(output_logits, label_tensor)
          loss_individual = self.individual_loss_fn(output_logits, label_tensor)

          # update the loss&accuracy during training
          avg_loss_metric_train.update(loss_val.item())
          accuracy_metric_train.update(output_probs, label_tensor)
          # update the entropy of samples cross iterations
          entropy_list.append(Categorical(logits=output_logits).entropy())    # calculate the entropy of samples at this iter
          id_list.append(id)                                                  # record the id order of samples at this iter
          all_individualLoss_list.append(loss_individual)
          # class_list.append(label_tensor)

          self.optimizer.zero_grad()
          if self.accumulation_steps:
            loss_val = loss_val / self.accumulation_steps
            loss_val.backward()
            if ((batch_id + 1) % self.accumulation_steps == 0) or (batch_id +1 == len(self.dataloader['train'])):
                self.optimizer.step()
                # self.optimizer.zero_grad()
          else:
            loss_val.backward()
            self.optimizer.step()
            # self.optimizer.zero_grad()


      else:
        if (self.AugmentedDataset_func !=1) and epoch in self.change_augmente_epochs_list:
          augmented_img_list = torch.tensor([])
          augmented_label_list = torch.tensor([], dtype=torch.long)
          augmented_id_list = torch.tensor([], dtype=torch.long)

        for batch_id, (img_tensor, label_tensor, id) in enumerate(train_dataloader):
          self.model.train()

          if (self.AugmentedDataset_func !=1) and epoch in self.change_augmente_epochs_list:
            # transformed_images.append((img_tensor, label_tensor, id))
            augmented_img_list = torch.cat([augmented_img_list, img_tensor], dim= 0)
            augmented_label_list = torch.cat([augmented_label_list, label_tensor], dim= 0)
            augmented_id_list = torch.cat([augmented_id_list, id], dim= 0)

          img_tensor = Variable(img_tensor).to(device)
          label_tensor = Variable(label_tensor).to(device)
          output_logits = self.model(img_tensor)

          output_probs = torch.nn.Softmax(dim=1)(output_logits)
          loss_val = self.loss_fn(output_logits, label_tensor)
          loss_individual = self.individual_loss_fn(output_logits, label_tensor)

          # update the loss&accuracy during training
          avg_loss_metric_train.update(loss_val.item())
          accuracy_metric_train.update(output_probs, label_tensor)
          # update the entropy of samples cross iterations
          entropy_list.append(Categorical(logits=output_logits).entropy())    # calculate the entropy of samples at this iter
          id_list.append(id)                                                  # record the id order of samples at this iter
          all_individualLoss_list.append(loss_individual)
          # class_list.append(label_tensor)

          self.optimizer.zero_grad()
          if self.accumulation_steps:
            loss_val = loss_val / self.accumulation_steps
            loss_val.backward()
            if ((batch_id + 1) % self.accumulation_steps == 0) or (batch_id +1 == len(train_dataloader)):
                self.optimizer.step()
                # self.optimizer.zero_grad()
          else:
            loss_val.backward()
            self.optimizer.step()
            # self.optimizer.zero_grad()

        # built_denoiser, only starts after 10th epoch
          # if epoch >= self.start_epoch:
          if self.augmentation_type == 'builtIn_denoiser':
              denoiser_output = self.builtin_denoise_model(img_tensor)
              self.model.eval()
              denoiser_resnet_output = self.model(denoiser_output)
              denoiser_loss = self.denoiser_loss(denoiser_resnet_output, label_tensor) # crossEntropyLoss
              if self.in_denoiseRecons_lossFlag:
                denoiser_loss = (1-self.inAug_lamda)*denoiser_loss + self.inAug_lamda*(torch.nn.MSELoss(size_average=False)(denoiser_output, img_tensor)/img_tensor.shape[0])
              self.denoiser_optimizer.zero_grad()
              denoiser_loss.backward()
              self.denoiser_optimizer.step()
              denoiserLoss_metric.update(denoiser_loss.item())
          if self.augmentation_type == 'builtIn_vae':
                (mean, logvar), vae_output = self.reset_vae(img_tensor)
                self.model.eval()
                vae_resnet_output = self.model(vae_output)
                vae_resnet_loss = self.denoiser_loss(vae_resnet_output, label_tensor)   # crossEntropyLoss
                if self.in_denoiseRecons_lossFlag:
                  vae_loss = self.reset_vae.reconstruction_loss(vae_output, img_tensor) + self.reset_vae.kl_divergence_loss(mean, logvar)
                  vae_resnet_loss = (1-self.inAug_lamda)*vae_resnet_loss + self.inAug_lamda*vae_loss
                self.resnet_vae_optimizer.zero_grad()
                vae_resnet_loss.backward()
                self.resnet_vae_optimizer.step()
                resnet_vae_metric.update(vae_resnet_loss.item())

      ############## end of iter #######################
      if (self.AugmentedDataset_func !=1) and epoch in self.change_augmente_epochs_list and self.augmentation_type:
        # augmented_images, augmented_labels, augmented_ids = zip(*transformed_images)
        # stacked_images = torch.cat([img_tensor for img_tensor, _, _ in augmented_images], dim=0)
        # stacked_labels = torch.stack(augmented_labels, dim=0)
        # stacked_ids = torch.stack(augmented_ids, dim=0)
        new_augmented_loader = torch.utils.data.TensorDataset(augmented_img_list, augmented_label_list, augmented_id_list)
        new_augmented_loader = torch.utils.data.DataLoader(new_augmented_loader, batch_size=self.batch_size, shuffle=True)

      if self.lr_scheduler_flag:
        self.lr_Scheduler.step()
        if self.augmentation_type == 'builtIn_vae' and (epoch >= self.start_epoch):
          # self.inVae_lrScheduler.step()
          pass
        elif self.augmentation_type == 'builtIn_denoiser' and (epoch >= self.start_epoch):
          self.indenoiser_lrSceduler.step()
      #################################################################
      # End of iteration -- running over once all data in the dataloader
      #################################################################
      writer.add_histogram('Entropy of all samples across the epoch', torch.tensor(list(flatten(entropy_list))), epoch+1)
      # visualization of output of in_denoiser
      if epoch >= self.start_epoch and self.augmentation_type:
        if self.augmentation_type == 'builtIn_denoiser' or (self.augmentation_type == 'builtIn_vae'):
          builtIn_modelComment = str(self.augmentation_type) + '/'
          if self.in_denoiseRecons_lossFlag:
            builtIn_modelComment += 'discr_reconstrLoss'
          else:
            builtIn_modelComment += 'discrLoss'
          if self.augmentation_type == 'builtIn_denoiser':
            writer.add_scalar(builtIn_modelComment, denoiserLoss_metric.compute().cpu().numpy(), epoch+1)
          else:
            writer.add_scalar(builtIn_modelComment, resnet_vae_metric.compute().cpu().numpy(), epoch+1)
      # writer.add_histogram('Loss of all samples across the epoch', torch.tensor(list(flatten(loss_candidates_list))), epoch+1)

      # start to collect the hard samples infos at the first epoch
      # history_candidates_id: storage all history candidates id cross epochs          //  if self.random_candidateSelection true, currentEpoch_candidateId are randomly choosed
      history_candidates_id, currentEpoch_lossCandidate, currentEpoch_candidateId, currentEpoch_candidateEnt = self.selection_candidates(current_allId_list=id_list, current_allEnt_list=entropy_list, current_allLoss_list=all_individualLoss_list,
                                                                                                history_candidates_id=history_candidates_id,
                                                                                                # history_entropy_candidates=history_entropy_candidates, history_num_candidates=history_num_candidates, history_meanLoss_candidates=history_meanLoss_candidates,
                                                                                                randomCandidate_selection=self.random_candidateSelection)

      writer.add_scalar('Number of hard samples/Train', len(currentEpoch_candidateId), epoch+1) # check the number of candidates at this epoch
      writer.add_scalar('Mean loss of hard samples/Train', np.mean(currentEpoch_lossCandidate), epoch+1)
      # if not self.augmentation_type:
      #   if len(currentEpoch_candidateId):
          # hard_image_display = self.dataloader['train'].dataset[list(currentEpoch_candidateId)[0]][0]
          # hard_class_display = self.dataloader['train'].dataset[list(currentEpoch_candidateId)[0]][1]
          # writer.add_image('Display/Hard sample', hard_image_display, epoch+1)
          # if len(currentEpoch_candidateId) > 1:
          #   hard_image_display_1 = self.dataloader['train'].dataset[list(currentEpoch_candidateId)[1]][0]
          #   hard_class_display_1 = self.dataloader['train'].dataset[list(currentEpoch_candidateId)[1]][1]          
          #   writer.add_image('Display/Hard sample_01', hard_image_display_1, epoch+1)
          #   writer.add_text('text id', str(hard_class_display) + str(hard_class_display_1), epoch+1)

      # if augmente
      if self.augmentation_type: # or self.builtin_denoise_flag

          # Augmentation_Method, if augmente at j_th; passing hardsamples infor to the augmentation_method()
          if epoch in self.augmente_epochs_list: # when current_epoch is at 10th, 20th, ..., 90th epoch, augmentate the dataset
            if self.random_candidateSelection:
              augmemtation_id = currentEpoch_candidateId
            # choose the hard samples according to the cross-entropy
            else:
              if self.k_epoch_sampleSelection!=0:  # if you choose to use hard samples over previous k epochs, or use the lastest epoch's hard samples (self.k_epoch_sampleSelection=0)
                print(f'use previous {self.k_epoch_sampleSelection}epoch_sampleSelection')
                k_epoch_candidateId = self.commonId_k_epochSelection(history_candidates_id=history_candidates_id, k=self.k_epoch_sampleSelection)  # select the common candidates from the last 3 epochs
                if len(k_epoch_candidateId) != 0:
                  augmemtation_id = k_epoch_candidateId
                  print(f"{len(augmemtation_id)} common_ids at epoch {epoch+1}")
                else:
                  augmemtation_id = currentEpoch_candidateId   # if randomSelection, or just use current current
              else: # if not choose hard samples over previous k epochs, then choose the hard samples at this epoch
                print('use current_epoch_sampleSelection')
                augmemtation_id = currentEpoch_candidateId

            if list(augmemtation_id):
              print(f"did augmentation at {epoch+1} epoch")
              ######################################################################################
              #### augMethod1
              ######################################################################################
              if self.AugmentedDataset_func == 1:
              # remain the same augmented dataset for the next 10 epochs
                augmented_dataset.augmentation_type = self.augmentation_type
                augmented_dataset.target_idx_list = list(augmemtation_id)
                augmented_dataset.tensorboard_epoch = epoch+1
                augmented_dataset.tf_writer = writer
                augmented_dataset.residual_connection_flag=self.residual_connection_flag
                augmented_dataset.residual_connection_method=self.residual_connection_method
                # denoiser for vae model
                augmented_dataset.denoise_flag=self.denoise_flag
                augmented_dataset.denoise_model=self.denoise_model
                # built-in denoiser
                augmented_dataset.builtIn_denoise_model = self.builtin_denoise_model
                augmented_dataset.in_denoiseRecons_lossFlag = self.in_denoiseRecons_lossFlag
                # built-in vae
                augmented_dataset.builtIn_vae_model = self.builtin_denoise_model
              ######################################################################################
              #### augMethod2
              ######################################################################################
              elif self.AugmentedDataset_func == 2:
                current_dataset = AugmentedDataset2(
                  dataset = train_dataloader.dataset,
                  target_idx_list = list(augmemtation_id),
                  augmentation_type = self.augmentation_type,
                  augmentation_transforms = self.augmentation_transforms,
                  model = self.augmentation_model,
                  model_transforms = self.model_transforms,
                  tensorboard_epoch = epoch+1,
                  tf_writer = writer,
                  residual_connection_flag = self.residual_connection_flag,
                  residual_connection_method=self.residual_connection_method,
                  denoise_flag=self.denoise_flag,
                  denoise_model=self.denoise_model,
                  builtIn_denoise_model = self.builtin_denoise_model,
                  in_denoiseRecons_lossFlag = self.in_denoiseRecons_lossFlag,
                  builtIn_vae_model = self.builtin_denoise_model,
                )
                train_dataloader = torch.utils.data.DataLoader(current_dataset, batch_size=self.batch_size, shuffle=True)
                # test_dataloader =  torch.utils.data.DataLoader(create_augmented_dataloader(train_dataloader), batch_size=self.batch_size, shuffle=False)
                # print('测试',len(test_dataloader.dataset[0]))
                # test_loader = torch.utils.data.DataLoader(CustomDataset(train_dataloader.dataset), batch_size=self.batch_size, shuffle=False)

              elif self.AugmentedDataset_func == 3:
                current_dataset = AugmentedDataset2(
                  dataset = self.dataloader['train'].dataset,
                  target_idx_list = list(augmemtation_id),
                  augmentation_type = self.augmentation_type,
                  augmentation_transforms = self.augmentation_transforms,
                  model = self.augmentation_model,
                  model_transforms = self.model_transforms,
                  tensorboard_epoch = epoch+1,
                  tf_writer = writer,
                  residual_connection_flag = self.residual_connection_flag,
                  residual_connection_method=self.residual_connection_method,
                  denoise_flag=self.denoise_flag,
                  denoise_model=self.denoise_model,
                  builtIn_denoise_model = self.builtin_denoise_model,
                  in_denoiseRecons_lossFlag = self.in_denoiseRecons_lossFlag,
                  builtIn_vae_model = self.builtin_denoise_model,
                )
                train_dataloader = torch.utils.data.DataLoader(current_dataset, batch_size=self.batch_size, shuffle=True)

            else:
              print(f'no augmentation at {epoch} epoch as there are no hard samples')

            # print(f"epoch {epoch} and its target_idx_list is {list(train_dataloader.dataset.target_idx_list)}")
          elif epoch in self.change_augmente_epochs_list:
            if self.AugmentedDataset_func == 2 or self.AugmentedDataset_func == 3:
              current_dataset.target_idx_list = []
              train_dataloader = new_augmented_loader



          if (epoch>=self.augmente_epochs_list[0]) and (self.random_candidateSelection is False) and (self.k_epoch_sampleSelection != 0):
            if (self.k_epoch_sampleSelection!=0) and (augmemtation_id):
              search_ids = torch.tensor(list(augmemtation_id))                    # common_Id from k previous epochs
              searchRange_ids = torch.tensor(currentEpoch_candidateId)              # id of candidates at this epoch
              loss_allcandidates = np.asarray(currentEpoch_lossCandidate).tolist()  # loss of candidates at this epoch
              common_id_indices = torch.hstack([torch.where(searchRange_ids == id_search)[0] for id_search in search_ids]).tolist()  # get the indices of common_id in the searchRange_ids
              common_id_loss = [loss_allcandidates[i] for i in common_id_indices]
              print(f"k_epoch_common_hardSamples mean loss {np.mean(common_id_loss)}")
              writer.add_scalar('Mean loss of k_epoch_common_hardSamples', np.mean(common_id_loss), epoch+1)


      self.model.eval()
      with torch.no_grad():
        for i, (img_tensor, label_tensor, idx) in enumerate(self.dataloader['valid']):
          img_tensor = img_tensor.to(device)
          label_tensor = label_tensor.to(device)
          valid_output_logits = self.model(img_tensor)
          valid_output_probs = torch.nn.Softmax(dim=1)(valid_output_logits)
          loss_val = self.loss_fn(valid_output_logits, label_tensor)
          loss_val_individual = self.individual_loss_fn(valid_output_logits, label_tensor)
          avg_loss_metric_test.update(loss_val.item())
          accuracy_metric_test.update(valid_output_probs, label_tensor)


          valid_entropy_list.append(Categorical(logits=valid_output_logits).entropy())    # calculate the entropy of samples at this iter
          valid_id_list.append(id)                                                        # record the id order of samples at this iter
          valid_all_individualLoss_list.append(loss_val_individual)


        valid_history_candidates_id, valid_currentEpoch_lossCandidate, valid_currentEpoch_candidateId, valid_currentEpoch_candidateEnt = self.selection_candidates(
                                                                                                current_allId_list = valid_id_list,
                                                                                                current_allEnt_list= valid_entropy_list,
                                                                                                current_allLoss_list= valid_all_individualLoss_list,
                                                                                                history_candidates_id= valid_history_candidates_id,
                                                                                                # history_entropy_candidates=history_entropy_candidates, history_num_candidates=history_num_candidates, history_meanLoss_candidates=history_meanLoss_candidates,
                                                                                                randomCandidate_selection=self.random_candidateSelection)

        writer.add_scalar('Number of hard samples/Valid', len(valid_currentEpoch_candidateId), epoch+1) # check the number of candidates at this epoch
        writer.add_scalar('Mean loss of hard samples/Valid', np.mean(valid_currentEpoch_lossCandidate), epoch+1)

      #######################################################################
      # end of one epoch
      #######################################################################
      average_loss_train_epoch = avg_loss_metric_train.compute().cpu().numpy()
      average_loss_test_epoch = avg_loss_metric_test.compute().cpu().numpy()
      average_accuracy_train_epoch = accuracy_metric_train.compute().item()
      average_accuracy_test_epoch = accuracy_metric_test.compute().item()

      print('Epoch[{}/{}]: loss_train={:.4f}, loss_test={:.4f},  accuracy_train={:.3f}, accuracy_test={:.3f}'.format(epoch+1, self.run_epochs,
                                                                                                                    average_loss_train_epoch, average_loss_test_epoch,
                                                                                                                    average_accuracy_train_epoch, average_accuracy_test_epoch,
                                                                                                                    ), flush=True)
      writer.add_scalar('Loss/train',average_loss_train_epoch, epoch+1)           # train_loss[-1]
      writer.add_scalar('Loss/valid', average_loss_test_epoch, epoch+1)            # test_loss[-1]
      writer.add_scalar('Accuracy/train',average_accuracy_train_epoch, epoch+1)  # train_accuracy[-1]
      writer.add_scalar('Accuracy/valid', average_accuracy_test_epoch, epoch+1)    # test_accuracy[-1]

      avg_loss_metric_train.reset()
      accuracy_metric_train.reset()
      avg_loss_metric_test.reset()
      accuracy_metric_test.reset()

      denoiserLoss_metric.reset()
      resnet_vae_metric.reset()

    # test
    Avg_loss_metric_test = torchmetrics.MeanMetric().to(device)
    Accuracy_metric_test = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device)
    self.model.eval()
    with torch.no_grad():
      for i, (img_tensor, label_tensor, idx) in enumerate(self.dataloader['test']):
        img_tensor = img_tensor.to(device)
        label_tensor = label_tensor.to(device)
        test_output_logits = self.model(img_tensor)
        test_output_probs = torch.nn.Softmax(dim=1)(test_output_logits)
        test_loss = self.loss_fn(test_output_logits, label_tensor)
        Avg_loss_metric_test.update(test_loss.item())
        Accuracy_metric_test.update(test_output_probs, label_tensor)

        test_entropy_list.append(Categorical(logits=test_output_logits).entropy())    # calculate the entropy of samples at this iter
        test_id_list.append(id)                                                        # record the id order of samples at this iter
        test_all_individualLoss_list.append(loss_val_individual)

      # end of iter
      Avg_loss_Test = Avg_loss_metric_test.compute().cpu().numpy()
      Acc_Test = Accuracy_metric_test.compute().item()

      test_loss_allSamples = list(flatten(test_all_individualLoss_list))
      test_entropy_allSamples = list(flatten(test_entropy_list))
      test_id_allSamples = list(flatten(test_id_list))
      test_select_Candidates = [(idx,ent,loss) for (idx,ent,loss) in zip(test_id_allSamples,test_entropy_allSamples,test_loss_allSamples) if ent >= self.entropy_threshold]
      _, _, test_candidates_loss = list(zip(*test_select_Candidates))
      test_candidates_loss = [loss.cpu().detach().numpy() for loss in test_candidates_loss]


      writer.add_text('test/loss', str(Avg_loss_Test), epoch+1)
      writer.add_text('test/acc', str(Acc_Test), epoch+1)
      writer.add_text('test/numHard', str(len(test_candidates_loss)) , epoch+1)
      writer.add_text('test/meanLoss_hard', str(np.mean(test_candidates_loss)), epoch+1)
      print('Test_loss', Avg_loss_Test, 'Test_acc',Acc_Test)

    writer.close()