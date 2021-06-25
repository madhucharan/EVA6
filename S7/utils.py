import torch
import torchvision
from dataset import Cifar10SearchDataset
import numpy as np
from torchsummary import summary
import torch.nn.functional as F

def get_stats(trainloader):
  """
  Args:
      trainloader (trainloader): Original data with no preprocessing
  Returns:
      mean: per channel mean
      std: per channel std
  """
  train_data = trainloader.dataset.data

  print('[Train]')
  print(' - Numpy Shape:', train_data.shape)
  print(' - Tensor Shape:', train_data.shape)
  print(' - min:', np.min(train_data))
  print(' - max:', np.max(train_data))

  train_data = train_data / 255.0

  mean = np.mean(train_data, axis=tuple(range(train_data.ndim-1)))
  std = np.std(train_data, axis=tuple(range(train_data.ndim-1)))

  print(f'\nDataset Mean - {mean}')
  print(f'Dataset Std - {std} ')

  return([mean, std])


def get_train_loader(transform=None):
  """
  Args:
      transform (transform): Albumentations transform
  Returns:
      trainloader: DataLoader Object
  """
  if transform:
    trainset = Cifar10SearchDataset(transform=transform)
  else:
    trainset = Cifar10SearchDataset(root="~/data/cifar10", train=True, 
                                    download=True)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)
  return(trainloader)


def get_test_loader(transform=None):
  """
  Args:
      transform (transform): Albumentations transform
  Returns:
      testloader: DataLoader Object
  """
  if transform:
    testset = Cifar10SearchDataset(transform=transform, train=False)
  else:
    testset = Cifar10SearchDataset(train=False)
  testloader = torch.utils.data.DataLoader(testset, batch_size=128, 
                                         shuffle=False, num_workers=2)

  return(testloader)


def get_summary(model, device):
  """
  Args:
      model (torch.nn Model): Original data with no preprocessing
      device (str): cuda/CPU
  """
  print(summary(model, input_size=(3, 32, 32)))



def get_device():
  """
  Returns:
      device (str): device type
  """
  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # For reproducibility
  if cuda:
      torch.cuda.manual_seed(SEED)
  else:
    torch.manual_seed(SEED)

  return(device)
