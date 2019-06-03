'''
Adversarially Robust Generalization Just Requires More Unlabeled Data
NeurIPS 2019 submission

Helper functions for train and test
'''

import torch
import os


def test(epoch, net, train_loader, test_loader, device):
  '''Test function
  Report natural train/test accuracy, robust train/test accuracy, defense success rate
  Also save the model'''
  
  print('===test(epoch={})==='.format(epoch))
  net.eval()
  robust_test_correct = 0
  robust_test_total = 0
  robust_train_correct = 0
  robust_train_total = 0
  natural_test_correct = 0
  natural_test_total = 0
  natural_train_correct = 0
  natural_train_total = 0

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs, pert_inputs = net(inputs, targets)

      _, predicted = outputs.max(1)
      robust_test_total += targets.size(0)
      robust_test_correct += predicted.eq(targets).sum().item()

      outputs = net(inputs)
      _, predicted = outputs.max(1)
      natural_test_total += targets.size(0)
      natural_test_correct += predicted.eq(targets).sum().item()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs, pert_inputs = net(inputs, targets)

      _, predicted = outputs.max(1)
      robust_train_total += targets.size(0)
      robust_train_correct += predicted.eq(targets).sum().item()

      outputs = net(inputs)
      _, predicted = outputs.max(1)
      natural_train_total += targets.size(0)
      natural_train_correct += predicted.eq(targets).sum().item()

  natural_test_acc = 100. * natural_test_correct / natural_test_total
  natural_train_acc = 100. * natural_train_correct / natural_train_total
  robust_test_acc = 100. * robust_test_correct / robust_test_total
  robust_train_acc = 100. * robust_train_correct / robust_train_total

  print('Natural train accuracy: {}'.format(natural_train_acc))
  print('Natural test accuracy: {}'.format(natural_test_acc))
  print('Robust train accuracy: {}'.format(robust_train_acc))
  print('Robust test accuracy: {}'.format(robust_test_acc))
  print('Defense success rate: {}'.format(robust_test_acc / natural_test_acc))
  
  # Save the model
  try:
    if not os.path.isdir('checkpoint'):
      os.makedirs('checkpoint')
    state = {
      'net': net.state_dict(),
      'epoch': epoch,
    }
    torch.save(state, 'checkpoint/{}.t7'.format(epoch))
  except OSError:
    print('OSError while saving {}.t7'.format(epoch))
    print('Ignoring...')


def pass_train(epoch, net, labeled_loader, unlabeled_loader, lbd, device, criterion, optimizer, scheduler):
  '''PASS algorithm'''

  print('===train(epoch={})==='.format(epoch))
  scheduler.step()
  net.train()

  correct_labeled = 0
  total_labeled = 0
  correct_unlabeled = 0
  total_unlabeled = 0

  unlabeled_enum = enumerate(unlabeled_loader)
  for batch_idx_labeled, (inputs_labeled, targets_labeled) in enumerate(labeled_loader):
    batch_idx_unlabeled, (inputs_unlabeled, _) = next(unlabeled_enum)
    inputs_labeled, targets_labeled = inputs_labeled.to(device), targets_labeled.to(device)
    inputs_unlabeled = inputs_unlabeled.to(device)

    prediction_unlabeled = net(inputs_unlabeled)  # Get prediction on unlabeled data
    targets_unlabeled = torch.argmax(prediction_unlabeled, dim=1)  # Use them as targets

    optimizer.zero_grad()
    outputs_labeled, _ = net(inputs_labeled, targets_labeled)
    outputs_unlabeled, _ = net(inputs_unlabeled, targets_unlabeled)
    loss_labeled = criterion(outputs_labeled, targets_labeled)
    loss_unlabeled = criterion(outputs_unlabeled, targets_unlabeled)
    loss = loss_labeled + lbd * loss_unlabeled
    loss.backward()
    optimizer.step()

    _, predicted = outputs_labeled.max(1)
    total_labeled += predicted.size(0)
    correct_labeled += predicted.eq(targets_labeled).sum().item()
    _, predicted = outputs_unlabeled.max(1)
    total_unlabeled += predicted.size(0)
    correct_unlabeled += predicted.eq(targets_unlabeled).sum().item()

  # Print online accuracy
  print('Labeled acc: {}\tUnlabeled acc: {}'.
        format(100. * correct_labeled / total_labeled,
               100. * correct_unlabeled / total_unlabeled))


def adv_train(epoch, net, traindata_loader, device, criterion, optimizer, scheduler):
  '''Original adversarial training'''

  print('===train(epoch={})==='.format(epoch))
  scheduler.step()
  net.train()

  for batch_idx, (inputs, targets) in enumerate(traindata_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs, pert_inputs = net(inputs, targets)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
