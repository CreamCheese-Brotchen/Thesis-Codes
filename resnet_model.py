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

from augmentation_folder.dataset_loader import IndexDataset, create_dataloaders
from augmentation_folder.augmentation_methods import simpleAugmentation_selection, AugmentedDataset, vae_augmentation
from VAE_folder.VAE_model import VAE
import random
# from memory_profiler import profile
# import sys 



class Resnet_trainer():
  def __init__(self, dataloader, num_classes, entropy_threshold, run_epochs, start_epoch, model, 
               loss_fn, individual_loss_fn, optimizer, tensorboard_comment, 
               augmentation_type=None, augmentation_transforms=None, 
               augmentation_model=None, model_transforms=None, 
               lr=0.001, l2=0, batch_size=64, accumulation_steps=2, 
               k_epoch_sampleSelection=3,
               random_candidateSelection=None,
               augmente_epochs_list=None):
    self.dataloader = dataloader
    self.entropy_threshold = entropy_threshold
    self.run_epochs = run_epochs
    self.start_epoch = start_epoch
    self.model = model
    self.lr = lr
    self.l2 = l2
    self.loss_fn = loss_fn      # torch.nn.CrossEntropyLoss()
    self.individual_loss_fn = individual_loss_fn
    self.optimizer = optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.l2) # torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)
    self.tensorboard_comment = tensorboard_comment
    self.augmentation_type = augmentation_type
    self.augmentation_transforms = augmentation_transforms
    self.augmentation_model = augmentation_model
    self.model_transforms = model_transforms
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.accumulation_steps  = accumulation_steps # gradient accumulation steps
    self.augmente_epochs_list = augmente_epochs_list  # number of epochs for augmentation, list [20, 30, ...., 90]
    self.k_epoch_sampleSelection = k_epoch_sampleSelection
    self.random_candidateSelection = random_candidateSelection
     

  def selection_candidates(self, current_allId_list, current_allEnt_list, current_allLoss_list, history_candidates_id, 
                           history_entropy_candidates, history_num_candidates, history_meanLoss_candidates,
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
        _type_: update the history of candidates: id, entropy, number, mean loss, which are collected across  epochs
    """
    idList_allSamples = list(flatten(current_allId_list))
    entropy_allSamples = list(flatten(current_allEnt_list))   # flattened, a tensor list of entropy of candidates over batchs
    loss_allSamples = list(flatten(current_allLoss_list))
    if randomCandidate_selection:
      combined_info = list(zip(idList_allSamples, entropy_allSamples, loss_allSamples))
      np.random.shuffle(combined_info)
      random_candidateNum = int(np.random.randint(0.15*len(idList_allSamples), 0.75*len(idList_allSamples)))  # num random_candidates
      select_Candidates = combined_info[:random_candidateNum]
    else:
      select_Candidates = [(idx,ent,loss) for (idx,ent,loss) in zip(idList_allSamples,entropy_allSamples,loss_allSamples) if ent >= self.entropy_threshold]     

    # select_Candidates = [(idx,ent) for (idx,ent) in zip(idList_allSamples,entropy_allSamples) if ent >= self.entropy_threshold] 
    # candidates_id, candidates_entropy = list(zip(*select_Candidates)) 
    if len(select_Candidates):     
      candidates_id, candidates_entropy, candidates_loss = list(zip(*select_Candidates))                 
      history_candidates_id.append(candidates_id)     # to search the past k epochs for searching more stubborn candidates
      # history_num_candidates.append(len(candidates_id))
      # currentEpoch_entroyCandidates = candidates_entropy
    #   history_entropy_candidates.append(candidates_entropy)
      candidates_loss_cpu = [loss.cpu().detach().numpy() for loss in candidates_loss]
      currentEpoch_lossCandidate = np.mean(candidates_loss_cpu)
    else:
      candidates_id = []
    #   print('No candidates at this epoch')
      history_candidates_id.append([])
      # history_num_candidates.append(0)
      # history_meanLoss_candidates.append(0)
    #   history_entropy_candidates.append(0)
      currentEpoch_lossCandidate = np.nan

    return history_candidates_id, currentEpoch_lossCandidate, candidates_id
    # return history_num_candidates, history_meanLoss_candidates, history_candidates_id, history_entropy_candidates, candidates_id


  def commonId_k_epochSelection(self, history_candidates_id, k):
    lists_dict = {}
    if len(history_candidates_id) < k:
      previousEpoch_selection = len(history_candidates_id)
    else:
      previousEpoch_selection = k
    for i in range(previousEpoch_selection):
      list_name = f"list_{i}"
      lists_dict[list_name] = set(history_candidates_id[-(i+1)])

    common_id = None
    for j in range(len(lists_dict)):
      t1 = list(lists_dict.items())[j:]
      common_id = set.intersection(*dict(t1).values())
      # print(j, common_id)
      if common_id:
        break    
    return common_id
      
#   @profile
  def train(self):
    # basic train/test loss/accuracy
    train_loss = list()
    test_loss= list()
    train_accuracy = list()
    test_accuracy = list()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)
    self.model.train()

    # basic train/test loss/accuracy
    avg_loss_metric_train = torchmetrics.MeanMetric().to(device)
    accuracy_metric_train = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device)
    avg_loss_metric_test = torchmetrics.MeanMetric().to(device)
    accuracy_metric_test = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device)

    # define the training dataset
    augmented_dataset = AugmentedDataset(
        dataset = self.dataloader['train'].dataset,
        target_idx_list = [],
        augmentation_type = self.augmentation_type,
        augmentation_transforms = self.augmentation_transforms,
        model = self.augmentation_model,
        model_transforms = self.model_transforms,
        tensorboard_epoch = [],  # NEW
        tf_writer = []
        )
    
    train_dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=self.batch_size, shuffle=True)

    # tensorboard
    writer = SummaryWriter(comment=self.tensorboard_comment)
    print(f"Starting Training Run. Using device: {device}", flush=True)

    for epoch in range(self.run_epochs):

      id_list = list()                  # id of all samples at each epoch, cleaned when the new epoch starts
      entropy_list = list()             # entropy of all samples at each epoch, histogram, cleaned when the new epoch starts
      all_individualLoss_list = list()  # loss of all samples at each epoch
      
      # history id/num/entropy of candidates(hard samples)
      history_candidates_id = list()
      history_num_candidates = list()
      history_entropy_candidates = list()
      history_accuracy_candidates = list()
      history_meanLoss_candidates = list()

      for batch_id, (img_tensor, label_tensor, id) in enumerate(train_dataloader):  # changes in train_dataloader
        img_tensor = Variable(img_tensor).to(device)
        label_tensor = Variable(label_tensor).to(device)
        output_logits = self.model(img_tensor)
        loss_val = self.loss_fn(output_logits, label_tensor)
        loss_individual = self.individual_loss_fn(output_logits, label_tensor)

        # update the loss&accuracy during training
        avg_loss_metric_train.update(loss_val)
        accuracy_metric_train.update(output_logits, label_tensor)
        # update the entropy of samples cross iterations
        entropy_list.append(Categorical(logits=output_logits).entropy())    # calculate the entropy of samples at this iter
        id_list.append(id)                                                  # record the id order of samples at this iter
        all_individualLoss_list.append(loss_individual)

        if self.accumulation_steps:
          loss_val = loss_val / self.accumulation_steps
          loss_val.backward()
          if ((batch_id + 1) % self.accumulation_steps == 0) or (batch_id +1 == len(train_dataloader)):
              # print('performing gradient update')
              self.optimizer.step()
              self.optimizer.zero_grad()
        else:
          loss_val.backward()
          self.optimizer.step()
          self.optimizer.zero_grad()

      # End of iteration -- running over once all data in the dataloader
      writer.add_histogram('Entropy of all samples across the epoch', torch.tensor(list(flatten(entropy_list))), epoch+1)
      # writer.add_histogram('Loss of all samples across the epoch', torch.tensor(list(flatten(loss_candidates_list))), epoch+1)

      # start to collect the hard samples infos at the first epoch
      # history_candidates_id, history_entropy_candidates, candidates_id                                                                                          # current_allId_list, current_allEnt_list, current_allLoss_list
      history_candidates_id, currentEpoch_lossCandidate, currentEpoch_candidateId = self.selection_candidates(current_allId_list=id_list, current_allEnt_list=entropy_list, current_allLoss_list=all_individualLoss_list,
                                                                                              # history_candidates_id, history_entropy_candidates
                                                                                                history_candidates_id=history_candidates_id, history_entropy_candidates=history_entropy_candidates,
                                                                                              # history_num_candidates, history_meanLoss_candidates
                                                                                                history_num_candidates=history_num_candidates, history_meanLoss_candidates=history_meanLoss_candidates,
                                                                                                randomCandidate_selection=self.random_candidateSelection)
      # but only start to add them to the tensorboard at the start_epoch
      if epoch >= self.start_epoch:
        # print(f'{len(currentEpoch_candidateId)} candidates at epoch {epoch+1}')
        writer.add_scalar('Number of hard samples', len(currentEpoch_candidateId), epoch+1) # check the number of candidates at this epoch
        writer.add_scalar('Mean loss of hard samples', currentEpoch_lossCandidate, epoch+1)
      
      # if augmente
      if self.augmentation_type:
          if self.run_epochs > 20:
            if self.augmente_epochs_list is None:     # generate an epoch_list for augmentation
              self.augmente_epochs_list =  np.arange(self.start_epoch, self.run_epochs, 10)  # every 10 epochs (20, 30, ..., 90), augment the dataset 
          else:
            self.augmente_epochs_list =  np.arange(self.start_epoch, self.run_epochs, 2)  # debug mode, every 2 epochs, augment the dataset
          
          # if augmente at th
          if epoch in self.augmente_epochs_list: # when current_epoch is at 10th, 20th, ..., 90th epoch, augmentate the dataset
            if self.random_candidateSelection:
              augmemtation_id = currentEpoch_candidateId
            else:
              if self.k_epoch_sampleSelection:
                k_epoch_candidateId = self.commonId_k_epochSelection(history_candidates_id=history_candidates_id, k=self.k_epoch_sampleSelection)  # select the common candidates from the last 3 epochs
                if len(k_epoch_candidateId) != 0:
                  augmemtation_id = k_epoch_candidateId
                  print(f"{len(augmemtation_id)} common_ids at epoch {epoch+1}")
                else:
                  augmemtation_id = currentEpoch_candidateId
                  # print(f"no common_id in the previous k epochs")
              else: # if not choose hard samples over previous k epochs, then choose the hard samples at this epoch
                augmemtation_id = currentEpoch_candidateId

            
            # remain the same augmented dataset for the next 10 epochs

            augmented_dataset.augmentation_type = self.augmentation_type
            augmented_dataset.target_idx_list = list(augmemtation_id)
            augmented_dataset.tensorboard_epoch = epoch+1
            augmented_dataset.tf_writer = writer
            print(f"did augmentation at {epoch+1} epoch")
        #   else:
        #     print(f"no augmentation at {epoch} epoch")

            # print(f"epoch {epoch} and its target_idx_list is {list(train_dataloader.dataset.target_idx_list)}")


      self.model.eval()
      for i, (img_tensor, label_tensor, idx) in enumerate(self.dataloader['test']):
        img_tensor = img_tensor.to(device)
        label_tensor = label_tensor.to(device)
        test_output_logits = self.model(img_tensor)
        loss_val = self.loss_fn(test_output_logits, label_tensor)
        avg_loss_metric_test.update(loss_val)
        accuracy_metric_test.update(test_output_logits, label_tensor)
        # print((output_logits.argmax(dim=-1) == label_tensor).sum()) was test to ensure that accuracy calc is accurate


      train_loss.append(avg_loss_metric_train.compute().cpu().numpy()) # save the loss of each epoch
    #   print('memory avg_loss_metric_Ptrain', sys.getsizeof(avg_loss_metric_train), "at epoch ", epoch+1)
      test_loss.append(avg_loss_metric_test.compute().cpu().numpy())
      train_accuracy.append(accuracy_metric_train.compute().cpu().numpy())
      # print('len(train accuracy) per epoch ', len(train_accuracy), train_accuracy[-1])
      test_accuracy.append(accuracy_metric_test.compute().cpu().numpy())
      print('Epoch[{}/{}]: loss_train={:.4f}, loss_test={:.4f},  accuracy_train={:.3f}, accuracy_test={:.3f}'.format(epoch+1, self.run_epochs,
                                                                                                                    train_loss[-1], test_loss[-1],
                                                                                                                    train_accuracy[-1], test_accuracy[-1]), flush=True)
      writer.add_scalar('Loss/train',train_loss[-1], epoch+1)
      writer.add_scalar('Loss/test', test_loss[-1], epoch+1)
      writer.add_scalar('Accuracy/train', train_accuracy[-1], epoch+1)
      writer.add_scalar('Accuracy/test', test_accuracy[-1], epoch+1)
    
      avg_loss_metric_train.reset()
      accuracy_metric_train.reset()
      avg_loss_metric_test.reset()
      accuracy_metric_test.reset()

    writer.close()