'''
Adversarially Robust Generalization Just Requires More Unlabeled Data
NeurIPS 2019 submission

For adversarial training on cifar-10, we will use 10x wide ResNet-32, as in [3].

References:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
[3] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu.
Towards Deep Learning Models Resistant to Adversarial Attacks. In ICLR, 2018.

Acknowledgements:
[1] https://github.com/MadryLab/cifar10_challenge
[2] https://github.com/karandwivedi42/adversarial
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
  " 3x3 convolution with padding "
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet_Cifar(nn.Module):

  def __init__(self, block, layers, width=1, num_classes=10):
    super(ResNet_Cifar, self).__init__()
    self.inplanes = 16
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = self._make_layer(block, 16 * width, layers[0])
    self.layer2 = self._make_layer(block, 32 * width, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 64 * width, layers[2], stride=2)
    self.avgpool = nn.AvgPool2d(8, stride=1)
    self.fc = nn.Linear(64 * block.expansion * width, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion)
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet32_w10(**kwargs):
  model = ResNet_Cifar(BasicBlock, [5, 5, 5], width=10, **kwargs)
  return model


class AttackPGD(nn.Module):
  def __init__(self, basic_net, config):
    super(AttackPGD, self).__init__()
    self.basic_net = basic_net
    self.rand = config['random_start']
    self.init_step_size = config['step_size']
    self.step_size = config['step_size']
    self.epsilon = config['epsilon']
    self.init_num_steps = config['num_steps']
    self.num_steps = config['num_steps']
    self.up = config['up']
    self.down = config['down']
    assert config['loss_func'] == 'xent', 'Only xent supported for now.'

  def set_attack(self, step_size=0.0, num_steps=0):
    if step_size == 0.0:
      self.step_size = self.init_step_size
    else:
      self.step_size = step_size
    if num_steps == 0:
      self.num_steps = self.init_num_steps
    else:
      self.num_steps = num_steps

  def forward(self, inputs, targets=None):
    # if not args.attack:
    # return self.basic_net(inputs), inputs
    if not targets is None:
      x = inputs.detach()
      if self.rand:
        # x = x + torch_cifar.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x = x + torch.clamp(torch.zeros_like(x).normal_(0, self.epsilon / 4),
                            -self.epsilon / 2, self.epsilon / 2)
        x = torch.clamp(x, self.down, self.up)
      for i in range(self.num_steps):
        x.requires_grad_()
        with torch.enable_grad():
          logits = self.basic_net(x)
          loss = F.cross_entropy(logits, targets, size_average=False)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + self.step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
        x = torch.clamp(x, self.down, self.up)

      return self.basic_net(x), x
    else:
      return self.basic_net(inputs)


def adv_train_net(basic_net, eps=8.0, step_size=2.0, step_num=7):
  '''Wrap a basic net with PGD attack
  >>> net = adv_train_net(basic_net)
  net(inputs) is natural prediction
  net(inputs, targets) is adversarial prediction
  '''

  mean = [0.4914, 0.4822, 0.4465]
  std = [0.2023, 0.1994, 0.2010]
  config = {
    'epsilon': eps / 255 / max(std),
    'num_steps': step_num,
    'step_size': step_size / 255 / max(std),
    'random_start': True,
    'loss_func': 'xent',
    'up': (1 - max(mean)) / max(std),
    'down': (0 - min(mean)) / max(std)
  }
  return AttackPGD(basic_net, config)
