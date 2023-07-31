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

from dataset_loader import IndexDataset, create_dataloaders
from augmentation_methods import simpleAugmentation_selection, AugmentedDataset, vae_augmentation
from VAE_model import VAE


def model_numClasses(dataset_name):
  tenClasses = ['CIFAR10', 'SVHN', 'MNIST', 'FashionMNIST']
  if dataset_name in tenClasses:
    classes_num = 10
  if dataset_name == 'Flowers102':
    classes_num = 102
  if dataset_name == 'Food101':
    classes_num = 101
  return classes_num



#################################################################################################################
#### Model 
#################################################################################################################
class Resnet_trainer():
  def __init__(self, dataloader, num_classes, entropy_threshold, run_epochs, start_epoch, model, loss_fn, individual_loss_fn, optimizer, tensorboard_comment, 
               augmentation_flag, augmentation_type, augmentation_transforms, 
               augmentation_model=None, model_transforms=None, 
               lr=0.001, l2=0, batch_size=64, accumulation_steps=20):
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
    self.augmentation_flag = augmentation_flag
    self.augmentation_type = augmentation_type
    self.augmentation_transforms = augmentation_transforms
    self.augmentation_model = augmentation_model
    self.model_transforms = model_transforms
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.accumulation_steps  = accumulation_steps # gradient accumulation steps
    

  def selection_candidates(self, current_allId_list, current_allEnt_list, current_allLoss_list, history_candidates_id, history_entropy_candidates, history_num_candidates, history_meanLoss_candidates):
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
    select_Candidates = [(idx,ent,loss) for (idx,ent,loss) in zip(idList_allSamples,entropy_allSamples,loss_allSamples) if ent >= self.entropy_threshold]     # search for the instances with Top k losses
    if len(select_Candidates):
      candidates_id, candidates_entropy, candidates_loss = list(zip(*select_Candidates))                       # get the indices of instances with Top K Loss
      history_candidates_id.append(candidates_id)
      history_num_candidates.append(len(candidates_id))
      print(f"{len(candidates_id)} candidates at this epoch")
      history_entropy_candidates.append(candidates_entropy)
      candidates_loss_cpu = [loss.cpu().detach().numpy() for loss in candidates_loss]
      history_meanLoss_candidates.append(np.mean(candidates_loss_cpu))
    else:
      candidates_id = []
      print('No candidates at this epoch')
      history_candidates_id.append([])
      history_num_candidates.append(0)
      history_meanLoss_candidates.append(0)
      history_entropy_candidates.append(0)

    return history_num_candidates, history_meanLoss_candidates, history_candidates_id, history_entropy_candidates, candidates_id


  def train(self):
    # basic train/test loss/accuracy
    train_loss = list()
    test_loss= list()
    train_accuracy = list()
    test_accuracy = list()

    # history id/num/entropy of candidates(hard samples)
    history_candidates_id = list()
    history_num_candidates = list()
    history_entropy_candidates = list()
    history_accuracy_candidates = list()
    history_meanLoss_candidates = list()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)
    self.model.train()

    # basic train/test loss/accuracy
    avg_loss_metric_train = torchmetrics.MeanMetric().to(device)
    accuracy_metric_train = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device)
    avg_loss_metric_test = torchmetrics.MeanMetric().to(device)
    accuracy_metric_test = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device)

    # define the training dataset
    print("creating augmented dataset")
    augmented_dataset = AugmentedDataset(
        dataset = self.dataloader['train'].dataset,
        target_idx_list = [],
        augmentation_type = self.augmentation_type,
        augmentation_transforms = self.augmentation_transforms,
        model = self.augmentation_model,
        model_transforms = self.model_transforms
        )
    
    train_dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=self.batch_size, shuffle=True)

    # tensorboard
    writer = SummaryWriter(comment=self.tensorboard_comment)
    print(f"Starting Training Run. Using device: {device}", flush=True)

    for epoch in range(self.run_epochs):

      id_list = list()              # id of all samples at each epoch, cleaned when the new epoch starts
      entropy_list = list()         # entropy of all samples at each epoch, histogram, cleaned when the new epoch starts
      loss_candidates_list = list() # loss of all samples at each epoch


      for i, (img_tensor, label_tensor, id) in enumerate(train_dataloader):  # changes in train_dataloader
        img_tensor = Variable(img_tensor).to(device)
        label_tensor = Variable(label_tensor).to(device)
        output_logits = self.model(img_tensor)
        loss_val = self.loss_fn(output_logits, label_tensor)
        loss_individual = self.individual_loss_fn(output_logits, label_tensor)

        loss_val.backward()
        if (i + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            print('perform accumulation step')

      #  Perform an optimization step for the remaining accumulated gradients (if any)
      if (i + 1) % self.accumulation_steps != 0:
        self.optimizer.step()
        self.optimizer.zero_grad()
        print('perform accumulation step for the remaining batch')

        # update the loss&accuracy during training
        avg_loss_metric_train.update(loss_val)
        accuracy_metric_train.update(output_logits, label_tensor)
        # update the entropy of samples cross iterations
        entropy_list.append(Categorical(logits=output_logits).entropy())    # calculate the entropy of samples at this iter
        id_list.append(id)                                                  # record the id order of samples at this iter
        loss_candidates_list.append(loss_individual)

      writer.add_histogram('Entropy of all samples across the epoch', torch.tensor(list(flatten(entropy_list))), epoch+1)
      writer.add_histogram('Loss of all samples across the epoch', torch.tensor(list(flatten(loss_candidates_list))), epoch+1)

      if epoch >= self.start_epoch:
        # update the history data of num/id/entropy_value of the candidates
                                                                                            # current_allId_list, current_allEnt_list, current_allLoss_list
        history_num_candidates, history_meanLoss_candidates, _, _, currentEpoch_candidateId = self.selection_candidates(current_allId_list=id_list, current_allEnt_list=entropy_list, current_allLoss_list=loss_candidates_list,
                                                                                            # history_candidates_id, history_entropy_candidates
                                                                                              history_candidates_id=history_candidates_id, history_entropy_candidates=history_entropy_candidates,
                                                                                            # history_num_candidates, history_meanLoss_candidates
                                                                                              history_num_candidates=history_num_candidates, history_meanLoss_candidates=history_meanLoss_candidates)

        # augmente the candidate samples
        # self.dataloader['train']
        if self.augmentation_flag:
            train_dataloader.dataset.target_idx_list = currentEpoch_candidateId
            print("Augmenting the hard samples during training")

        writer.add_scalar('Number of hard samples', history_num_candidates[-1], epoch+1)
        writer.add_scalar('Mean loss of hard samples', history_meanLoss_candidates[-1], epoch+1)

      self.model.eval()
      for i, (img_tensor, label_tensor, idx) in enumerate(self.dataloader['test']):
        img_tensor = img_tensor.to(device)
        label_tensor = label_tensor.to(device)
        output_logits = self.model(img_tensor)
        loss_val = self.loss_fn(output_logits, label_tensor)
        avg_loss_metric_test.update(loss_val)
        accuracy_metric_test.update(output_logits, label_tensor)
        # print((output_logits.argmax(dim=-1) == label_tensor).sum()) was test to ensure that accuracy calc is accurate


      train_loss.append(avg_loss_metric_train.compute().cpu().numpy())
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Resnet Training script')

  parser.add_argument('--dataset', type=str, default='MNIST', choices=("MNIST", "CIFAR10", "FashionMNIST", "SVHN", "Flowers102", "Food101"), help='Dataset name')
  parser.add_argument('--entropy_threshold', type=float, default=0.5, help='Entropy threshold')
  parser.add_argument('--run_epochs', type=int, default=5, help='Number of epochs to run')
  parser.add_argument('--candidate_start_epoch', type=int, default=0, help='Epoch to start selecting candidates. Candidate calculation begind after the mentioned epoch')
  parser.add_argument('--tensorboard_comment', type=str, default='test_run', help='Comment to append to tensorboard logs')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--l2', type=float, default=1e-4, help='L2 regularization')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
  parser.add_argument('--not_pretrained', action='store_true', help='Use randomly initialized weights instead of pretrained weights')
  parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')
  parser.add_argument('--augmentation_flag', action='store_true', help='Not augmenting the candidate samples')
  parser.add_argument('--augmentation_type', type=str, default=None, choices=("vae", "simple"), help='Augmentation type')
  parser.add_argument('--simpleAugmentaion_name', type=str, default=None, choices=("random_color", "center_crop", "gaussian_blur", "elastic_transform", "random_perspective", "random_resized_crop", "random_invert", "random_posterize", "rand_augment", "augmix"), help='Simple Augmentation name')
  parser.add_argument('--accumulation_steps', type=int, default=20, help='Number of accumulation steps')
  args = parser.parse_args()
  print(f"Script Arguments: {args}", flush=True)


  transforms_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
  ]
  )
  transforms_test = transforms.Compose([
      transforms.Grayscale(num_output_channels=3),
      transforms.Resize(256),
      transforms.CenterCrop(256),
      transforms.ToTensor()
  ]
  )

  dataset_loaders = create_dataloaders(transforms_train, transforms_test, args.batch_size, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)
  classes_num = model_numClasses(args.dataset)
  print(f"Number of classes: {classes_num}", flush=True)


  #####################################################
  weights = ResNet18_Weights.DEFAULT
  if args.not_pretrained:
    weights=None
    resnet = resnet18(weights=weights)
    print(f"Using weights: {'None' if weights is None else weights}", flush=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, classes_num)  
  else:
    resnet = resnet18(num_classes=classes_num, pretrained=False)

  if args.augmentation_flag:
    print("Do Augmentation in this experiment")
  #####################################################

  if args.augmentation_type == "vae":
    print('using vae augmentation')
    input_height = 256
    vae_model = VAE(input_height=input_height)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.reduce_dataset:
      vae_trainEpochs = 1
    else: 
      vae_trainEpochs = 60 
    vae_trainer = Trainer(max_epochs=vae_trainEpochs, accumulate_grad_batches=5, accelerator="auto", strategy="auto", devices="auto", enable_progress_bar=False)
    vae_trainer.tune(vae_model, dataset_loaders['train'])
    vae_trainer.fit(vae_model, dataset_loaders['train'])
    
    # train resnet with vae augmentation
    model_trainer = Resnet_trainer(dataloader=dataset_loaders, num_classes=classes_num, entropy_threshold=args.entropy_threshold, run_epochs=args.run_epochs, start_epoch=args.candidate_start_epoch, 
                                 model=resnet, loss_fn=torch.nn.CrossEntropyLoss(), individual_loss_fn=torch.nn.CrossEntropyLoss(reduction='none') ,optimizer= torch.optim.Adam, tensorboard_comment=args.tensorboard_comment, 
                                 augmentation_flag=args.augmentation_flag, augmentation_type=args.augmentation_type, 
                                 augmentation_transforms=vae_augmentation, 
                                 augmentation_model=vae_model, model_transforms=vae_trainer, # augmentation_model=vae, model_transforms=vae_trainer (pass a Trainer of VAE)
                                 lr=args.lr, l2=args.l2, batch_size=args.batch_size, accumulation_steps=args.accumulate_steps)
  else: 
    # when simple augmentation is wanted, apply the func simpleAugmentation_selection 
    simpleAugmentation_method = simpleAugmentation_selection(args.simpleAugmentaion_name)
    # augmentation_method: {simpleAugmentation_selection, }
    model_trainer = Resnet_trainer(dataloader=dataset_loaders, num_classes=classes_num, entropy_threshold=args.entropy_threshold, run_epochs=args.run_epochs, start_epoch=args.candidate_start_epoch, 
                                 model=resnet, loss_fn=torch.nn.CrossEntropyLoss(), individual_loss_fn=torch.nn.CrossEntropyLoss(reduction='none') ,optimizer= torch.optim.Adam, tensorboard_comment=args.tensorboard_comment, 
                                 augmentation_flag=args.augmentation_flag, augmentation_type=args.augmentation_type, 
                                 augmentation_transforms=simpleAugmentation_method, 
                                 lr=args.lr, l2=args.l2, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps)
  model_trainer.train()