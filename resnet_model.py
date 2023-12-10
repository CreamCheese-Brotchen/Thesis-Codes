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
from augmentation_folder.augmentation_methods import simpleAugmentation_selection, AugmentedDataset, vae_augmentation, DenoisingModel
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
              lr=0.0001, l2=0, batch_size=64, accumulation_steps=2, 
              k_epoch_sampleSelection=3,
              random_candidateSelection=False,
              augmente_epochs_list=None,
              residual_connection_flag=False, residual_connection_method=None,   # resConnect/Denoise for vae
              denoise_flag=False, denoise_model=None,                            # resConnect/Denoise for vae
              in_denoiseRecons_lossFlag=False):                                     # built-in denoisers
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
    # resConnect/Denoise for vae
    self.residual_connection_flag=residual_connection_flag
    self.residual_connection_method=residual_connection_method
    self.denoise_flag = denoise_flag
    self.denoise_model = denoise_model
    # builtin_denoise
    self.builtin_denoise_model = DenoisingModel()
    self.denoiser_optimizer = torch.optim.Adam(self.builtin_denoise_model.parameters(), lr=0.0001)
    self.denoiser_loss = torch.nn.CrossEntropyLoss()
    self.in_denoiseRecons_lossFlag = in_denoiseRecons_lossFlag
    # basic params
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.accumulation_steps  = accumulation_steps # gradient accumulation steps
    self.augmente_epochs_list = augmente_epochs_list  # number of epochs for augmentation, list [20, 30, ...., 90]
    self.k_epoch_sampleSelection = k_epoch_sampleSelection
    self.random_candidateSelection = random_candidateSelection
     
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
    if randomCandidate_selection:
      combined_info = list(zip(idList_allSamples, entropy_allSamples, loss_allSamples))
      np.random.shuffle(combined_info)
      random_candidateNum = int(np.random.randint(0.15*len(idList_allSamples), 0.75*len(idList_allSamples)))  # num random_candidates
      select_Candidates = combined_info[:random_candidateNum]
      print(f"randomly select {random_candidateNum} candidates at this epoch")
    else:
      select_Candidates = [(idx,ent,loss) for (idx,ent,loss) in zip(idList_allSamples,entropy_allSamples,loss_allSamples) if ent >= self.entropy_threshold]     

    # select_Candidates = [(idx,ent) for (idx,ent) in zip(idList_allSamples,entropy_allSamples) if ent >= self.entropy_threshold] 
    # candidates_id, candidates_entropy = list(zip(*select_Candidates)) 
    if len(select_Candidates):     
      candidates_id, candidates_entropy, candidates_loss = list(zip(*select_Candidates))                 
      history_candidates_id.append(candidates_id)     # to search the past k epochs for searching more stubborn candidates
      # history_num_candidates.append(len(candidates_id))
      # currentEpoch_entroyCandidates = candidates_entropy
      # history_entropy_candidates.append(candidates_entropy)
      candidates_loss_cpu = [loss.cpu().detach().numpy() for loss in candidates_loss]
      currentEpoch_lossCandidate = candidates_loss_cpu  #np.mean(candidates_loss_cpu)
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
      
#   @profile
  def train(self):
    # basic train/test loss/accuracy
    train_loss = list()
    test_loss= list()
    train_accuracy = list()
    test_accuracy = list()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)

    # basic train/test loss/accuracy
    avg_loss_metric_train = torchmetrics.MeanMetric().to(device)
    accuracy_metric_train = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device)
    avg_loss_metric_test = torchmetrics.MeanMetric().to(device)
    accuracy_metric_test = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device)

    denoiserLoss_metric = torchmetrics.MeanMetric().to(device)

    # define the training dataset
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
    # print('self.residual_connection_flag', self.residual_connection_flag)
    # print('self.residual_connection_method', self.residual_connection_method, 'type', type(self.residual_connection_method))

    train_dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=self.batch_size, shuffle=True)

    # tensorboard
    writer = SummaryWriter(comment=self.tensorboard_comment)
    print(f"Starting Training Run. Using device: {device}", flush=True)

    history_candidates_id = list()
    history_num_candidates = list()
    history_entropy_candidates = list()
    history_accuracy_candidates = list()
    history_meanLoss_candidates = list()

    for epoch in range(self.run_epochs):

      id_list = list()                  # id of all samples at each epoch, cleaned when the new epoch starts
      entropy_list = list()             # entropy of all samples at each epoch, histogram, cleaned when the new epoch starts
      all_individualLoss_list = list()  # loss of all samples at each epoch
      
      # history id/num/entropy of candidates(hard samples)
      # history_candidates_id = list()
      # history_num_candidates = list()
      # history_entropy_candidates = list()
      # history_accuracy_candidates = list()
      # history_meanLoss_candidates = list()

      self.model.train()  
      for batch_id, (img_tensor, label_tensor, id) in enumerate(train_dataloader):  # changes in train_dataloader
      # for batch_id, (img_tensor, label_tensor, id) in enumerate(self.dataloader['train']):
        self.model.train()  
        self.optimizer.zero_grad()
        img_tensor = Variable(img_tensor).to(device)
        label_tensor = Variable(label_tensor).to(device)
        output_logits = self.model(img_tensor)
        
        output_probs = torch.nn.Softmax(dim=1)(output_logits)
        loss_val = self.loss_fn(output_probs, label_tensor)
        loss_individual = self.individual_loss_fn(output_probs, label_tensor)

        # update the loss&accuracy during training
        avg_loss_metric_train.update(loss_val.item())
        accuracy_metric_train.update(output_probs, label_tensor)
        # update the entropy of samples cross iterations
        entropy_list.append(Categorical(logits=output_logits).entropy())    # calculate the entropy of samples at this iter
        id_list.append(id)                                                  # record the id order of samples at this iter
        all_individualLoss_list.append(loss_individual)

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
        if self.augmentation_type == 'builtIn_denoiser':
            print('did train_builtIn_denoiser')
            denoiser_output = self.builtin_denoise_model(img_tensor)
            self.model.eval()
            denoiser_resnet_output = self.model(denoiser_output)
            denoiser_loss = self.denoiser_loss(denoiser_resnet_output, label_tensor)
            if self.in_denoiseRecons_lossFlag:
              denoiser_loss = 0.5*denoiser_loss + 0.5*(torch.nn.MSELoss(size_average=False)(denoiser_output, img_tensor)/img_tensor.shape[0])
            print(f"denoiser_loss {denoiser_loss.item()}")
            self.denoiser_optimizer.zero_grad()
            denoiser_loss.backward()
            self.denoiser_optimizer.step()
            denoiserLoss_metric.update(denoiser_loss.item())

      #################################################################
      # End of iteration -- running over once all data in the dataloader
      #################################################################
      writer.add_histogram('Entropy of all samples across the epoch', torch.tensor(list(flatten(entropy_list))), epoch+1)
      if self.augmentation_type == 'builtIn_denoiser':
        builtIn_denoiserComment = 'builtIn_denoiser/'
        if self.in_denoiseRecons_lossFlag:
          builtIn_denoiserComment += 'discr_reconstrLoss'
        else:
          builtIn_denoiserComment += 'discrLoss'
        writer.add_scalar(builtIn_denoiserComment, denoiserLoss_metric.compute().cpu().numpy(), epoch+1)
      # writer.add_histogram('Loss of all samples across the epoch', torch.tensor(list(flatten(loss_candidates_list))), epoch+1)

      # start to collect the hard samples infos at the first epoch
      # history_candidates_id: storage all history candidates id cross epochs          //  if self.random_candidateSelection true, currentEpoch_candidateId are randomly choosed                                                              
      history_candidates_id, currentEpoch_lossCandidate, currentEpoch_candidateId = self.selection_candidates(current_allId_list=id_list, current_allEnt_list=entropy_list, current_allLoss_list=all_individualLoss_list,
                                                                                                history_candidates_id=history_candidates_id, 
                                                                                                # history_entropy_candidates=history_entropy_candidates, history_num_candidates=history_num_candidates, history_meanLoss_candidates=history_meanLoss_candidates,
                                                                                                randomCandidate_selection=self.random_candidateSelection)
      # but only start to add them to the tensorboard at the start_epoch
      # if epoch >= self.start_epoch:
        # print(f'{len(currentEpoch_candidateId)} candidates at epoch {epoch+1}')
      writer.add_scalar('Number of hard samples', len(currentEpoch_candidateId), epoch+1) # check the number of candidates at this epoch
      writer.add_scalar('Mean loss of hard samples', np.mean(currentEpoch_lossCandidate), epoch+1)
      
      # if augmente
      if self.augmentation_type: # or self.builtin_denoise_flag
          # design the aug_epo list
          if self.run_epochs > 20:
            if self.augmente_epochs_list is None:     # generate an epoch_list for augmentation
              self.augmente_epochs_list =  np.arange(self.start_epoch, self.run_epochs, 10)  # every 10 epochs (20, 30, ..., 90), augment the dataset 
          else:
            self.augmente_epochs_list =  np.arange(self.start_epoch, self.run_epochs, 2)  # debug mode, every 2 epochs, augment the dataset
          
            
          # Augmentation_Method, if augmente at j_th
          if epoch in self.augmente_epochs_list: # when current_epoch is at 10th, 20th, ..., 90th epoch, augmentate the dataset
            if self.random_candidateSelection:
              augmemtation_id = currentEpoch_candidateId
            # choose the hard samples according to the cross-entropy 
            else:
              if self.k_epoch_sampleSelection != 0:  # if you choose to use hard samples over previous k epochs, or use the lastest epoch's hard samples (self.k_epoch_sampleSelection=0)
                print(f'use previous {self.k_epoch_sampleSelection}epoch_sampleSelection')
                k_epoch_candidateId = self.commonId_k_epochSelection(history_candidates_id=history_candidates_id, k=self.k_epoch_sampleSelection)  # select the common candidates from the last 3 epochs
                if len(k_epoch_candidateId) != 0:
                  augmemtation_id = k_epoch_candidateId
                  print(f"{len(augmemtation_id)} common_ids at epoch {epoch+1}")
                else:
                  augmemtation_id = currentEpoch_candidateId
                  # print(f"no common_id in the previous k epochs")
              else: # if not choose hard samples over previous k epochs, then choose the hard samples at this epoch
                print('use current_epoch_sampleSelection')
                augmemtation_id = currentEpoch_candidateId

            if list(augmemtation_id):
              print(f"did augmentation at {epoch+1} epoch") 
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
              # to visualize the common id candidates' performance
              if self.random_candidateSelection:
                pass
              else:
                if self.k_epoch_sampleSelection != 0:
                  search_ids = torch.tensor(list(augmemtation_id))                    # common_Id from k previous epochs
                  searchRange_ids = torch.tensor(currentEpoch_candidateId)              # id of candidates at this epoch
                  loss_allcandidates = np.asarray(currentEpoch_lossCandidate).tolist()  # loss of candidates at this epoch
                  # print(search_ids)
                  # print(searchRange_ids)
                  # print(loss_allcandidates)
                  common_id_indices = torch.hstack([torch.where(searchRange_ids == id_search)[0] for id_search in search_ids]).tolist()  # get the indices of common_id in the searchRange_ids
                  common_id_loss = [loss_allcandidates[i] for i in common_id_indices]
                  print(f"k_epoch_common_hardSamples mean loss {np.mean(common_id_loss)}")
                  writer.add_scalar('Mean loss of k_epoch_common_hardSamples', np.mean(common_id_loss), epoch+1)
            # if list(augmemtation_id) is empty, no hard samples & no augmentation
            else:  
              print(f'no augmentation at {epoch} epoch as there are no hard samples')

            # print(f"epoch {epoch} and its target_idx_list is {list(train_dataloader.dataset.target_idx_list)}")


      self.model.eval()
      with torch.no_grad():
        for i, (img_tensor, label_tensor, idx) in enumerate(self.dataloader['test']):
          img_tensor = img_tensor.to(device)
          label_tensor = label_tensor.to(device)
          test_output_logits = self.model(img_tensor)
          test_output_probs = torch.nn.Softmax(dim=1)(test_output_logits)
          loss_val = self.loss_fn(test_output_probs, label_tensor)
          avg_loss_metric_test.update(loss_val.item())
          accuracy_metric_test.update(test_output_probs, label_tensor)

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
      writer.add_scalar('Loss/test', average_loss_test_epoch, epoch+1)            # test_loss[-1]
      writer.add_scalar('Accuracy/train',average_accuracy_train_epoch, epoch+1)  # train_accuracy[-1]
      writer.add_scalar('Accuracy/test', average_accuracy_test_epoch, epoch+1)    # test_accuracy[-1]
    
      avg_loss_metric_train.reset()
      accuracy_metric_train.reset()
      avg_loss_metric_test.reset()
      accuracy_metric_test.reset()

      denoiserLoss_metric.reset()

    writer.close()