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

#################################################################################################################
#### Data Loaders 
#################################################################################################################
# add an unique id to every sample in the dataset
class IndexDataset(Dataset):
    def __init__(self, Dataset):
        self.Dataset = Dataset
        
    def __getitem__(self, index):
        data, target = self.Dataset[index]
        
        return data, target, index

    def __len__(self):
        return len(self.Dataset)
    

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

def create_dataloaders(transforms_train, transforms_test, batch_size, dataset_name, add_idx, reduce_dataset=False):
  if dataset_name == 'MNIST':
    data_train = datasets.MNIST(root='data', train=True, download=True, transform=transforms_train)         
    data_test = datasets.MNIST(root='data', train=False, download=True, transform=transforms_test)

  if dataset_name == 'FashionMNIST':
    data_train = datasets.FashionMNIST(root = 'data', train = True, download=True, transform = transforms_train)         
    data_test = datasets.FashionMNIST(root = 'data', train = False, download=True, transform = transforms_test)

  if dataset_name == 'CIFAR10': 
    data_train = datasets.CIFAR10(root = 'data', train = True, download=True, transform = transforms_train)         
    data_test = datasets.CIFAR10(root = 'data', train = False, download=True, transform = transforms_test)
  
  if dataset_name == 'SVHN':
    data_train = datasets.SVHN(root = 'data', split='train', download=True, transform = transforms_train)
    data_test = datasets.SVHN(root = 'data', split='test', download=True, transform = transforms_test)

  # debug
  if reduce_dataset:
    data_train = data_utils.Subset(data_train, torch.arange(32))
    data_test = data_utils.Subset(data_test, torch.arange(32))
 
  # give an unique id to every sample in the dataset to track the hard samples
  if add_idx:
      data_train = IndexDataset(data_train)
      data_test = IndexDataset(data_test)

  dataset_loaders = {
          'train' : torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True),
          'test'  : torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)
          }

  return dataset_loaders


#################################################################################################################
#### Augmented Data forming 
#################################################################################################################
# get the dataset, target_idx_list (hard samples' id), augmentation transform. If the id of the sample is in the target_id_list, then apply the transform to this sample 
class augmentation(Dataset):
    def __init__(self, dataset, target_idx_list, augmentation_transforms):
        self.dataset = dataset
        self.target_idx_list = target_idx_list
        self.augmentation_transforms = augmentation_transforms

    def __getitem__(self, index):
        data, label, idx = self.dataset[index]

        # Apply data augmentation based on the index being in the target index list
        if idx in self.target_idx_list:
            data  = self.augmentation_transforms(data)

        return data, label, idx

    def __len__(self):
        return len(self.dataset)





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
    augmentation_method = transforms.Compose([transforms.ToPILImage(), transforms.RandomInvert(), transforms.ToTensor()])  
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
#### Model 
#################################################################################################################
class Trainer():
  def __init__(self, dataloader, entropy_threshold, run_epochs, start_epoch, model, loss_fn, individual_loss_fn, optimizer, tensorboard_comment, augmentation_transforms, lr=0.001, l2=0, batch_size=64):
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
    self.augmentation_transforms = augmentation_transforms
    self.batch_size = batch_size


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
    accuracy_metric_train = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    avg_loss_metric_test = torchmetrics.MeanMetric().to(device)
    accuracy_metric_test = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    # define the training dataset
    train_dataloader = copy.deepcopy(self.dataloader['train'])

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
        self.optimizer.zero_grad()
        output_logits = self.model(img_tensor)
        loss_val = self.loss_fn(output_logits, label_tensor)
        loss_individual = self.individual_loss_fn(output_logits, label_tensor)

        loss_val.backward()
        self.optimizer.step()

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
        if len(currentEpoch_candidateId):
          augmented_dataset = augmentation(self.dataloader['train'].dataset, currentEpoch_candidateId, self.augmentation_transforms)
          augmented_dataset = torch.utils.data.DataLoader(augmented_dataset, batch_size=self.batch_size, shuffle=True)
          train_dataloader = copy.deepcopy(augmented_dataset)

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

  parser.add_argument('--dataset', type=str, default='MNIST', choices=("MNIST", "CIFAR10", "FashionMNIST", "SVHN"), help='Dataset name')
  parser.add_argument('--entropy_threshold', type=float, default=0.5, help='Entropy threshold')
  parser.add_argument('--run_epochs', type=int, default=5, help='Number of epochs to run')
  parser.add_argument('--candidate_start_epoch', type=int, default=0, help='Epoch to start selecting candidates. Candidate calculation begind after the mentioned epoch')
  parser.add_argument('--tensorboard_comment', type=str, default='test_run', help='Comment to append to tensorboard logs')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--l2', type=float, default=1e-4, help='L2 regularization')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
  parser.add_argument('--not_pretrained', action='store_true', help='Use randomly initialized weights instead of pretrained weights')
  parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')
  parser.add_argument('--simpleAugmentaion_name', type=str, default=None, choices=("random_color", "center_crop", "gaussian_blur", "elastic_transform", "random_perspective", "random_resized_crop", "random_invert", "random_posterize", "rand_augment", "augmix"), help='Simple Augmentation name')

  args = parser.parse_args()
  print(f"Script Arguments: {args}", flush=True)

  dataset_loaders = create_dataloaders(transforms_train, transforms_test, args.batch_size, args.dataset, add_idx=True, reduce_dataset=args.reduce_dataset)
  
  weights = ResNet18_Weights.DEFAULT
  if args.not_pretrained:
    weights=None

  resnet = resnet18(weights=weights)
  print(f"Using weights: {'None' if weights is None else weights}", flush=True)
  num_ftrs = resnet.fc.in_features
  resnet.fc = torch.nn.Linear(num_ftrs, 10)
  augmentation_method = simpleAugmentation_selection(args.simpleAugmentaion_name)
  model_trainer = Trainer(dataloader=dataset_loaders, entropy_threshold=args.entropy_threshold, run_epochs=args.run_epochs, start_epoch=args.candidate_start_epoch, model=resnet, loss_fn=torch.nn.CrossEntropyLoss(), individual_loss_fn=torch.nn.CrossEntropyLoss(reduction='none') ,optimizer= torch.optim.Adam, tensorboard_comment=args.tensorboard_comment, augmentation_transforms=augmentation_method, lr=args.lr, l2=args.l2, batch_size=args.batch_size)
  model_trainer.train()