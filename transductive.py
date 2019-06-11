'''
Adversarially Robust Generalization Just Requires More Unlabeled Data
NeurIPS 2019 submission

PASS on Cifar-10: Transductive Setting

In the transductive setting, the test set is used as unlabeled data.
All labels in the test set will be masked out.
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import argparse
import numpy as np

import model
import utils


if __name__=='__main__':

  parser = argparse.ArgumentParser(description='PASS on Cifar-10: Transductive Setting')
  parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
  parser.add_argument('--lbd', default=0.125, type=float, help='weight of unlabeled data')
  args = parser.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Prepare data
  print('==> Preparing data..')
  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  labeled_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
  unlabeled_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
  unlabeled_set.targets = np.zeros((len(unlabeled_set.targets),))  # Mask all test set labels
  
  # In each mini-batch, there are 100 labeled points and 20 unlabeled points
  labeled_loader = torch.utils.data.DataLoader(labeled_set,
                                              batch_size=100,
                                              shuffle=True, num_workers=2)
  unlabeled_loader = torch.utils.data.DataLoader(unlabeled_set,
                                                batch_size=20,
                                                shuffle=True, num_workers=2)

  # Data loader used for test
  testdata_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
  testdata_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

  testdata_train_loader = torch.utils.data.DataLoader(testdata_train,
                                              batch_size=128,
                                              shuffle=True, num_workers=2)
  testdata_test_loader = torch.utils.data.DataLoader(testdata_test,
                                                batch_size=128,
                                                shuffle=True, num_workers=2)

  # Build model
  print('==> Building model..')
  basic_net = model.resnet32_w10()
  basic_net = basic_net.to(device)
  net = model.adv_train_net(basic_net)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  # Train
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
  scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)

  for epoch in range(401):
    utils.pass_train(epoch, net, labeled_loader, unlabeled_loader,
                     args.lbd, device, criterion, optimizer, scheduler)
    if epoch % 10 == 0:
      utils.test(epoch, net, testdata_train_loader, testdata_test_loader, device)
