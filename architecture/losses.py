from __future__ import absolute_import
import sys

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['MSE', 'Perceptual', 'CSCP', 'Adversarial']

class MSE(nn.Module):
  """
    Computes MSE loss between inputs
  """
  def __init__(self):
    super(MSE, self).__init__()
    self.mse = nn.MSELoss()
  def forward(self, inputs, targets):
    loss = 0
    if isinstance(inputs, list):
      for x, y in zip(inputs, targets):
        loss += self.mse(x, y)
    else:
      loss = self.mse(inputs, targets)
    return loss

class Perceptual(nn.Module):
  """
        Computes Perceptual loss between inputs
  """
  def __init__(self):
    super(Perceptual, self).__init__()
    vgg = torchvision.models.vgg.vgg16(pretrained=True)
    self.vgg_layers = vgg.features
    for name, layer in self.vgg_layers._modules.items():
      layer.require_grad = False

  def vgg_feature(self, x):
    # x has only one channel and vgg expects 3
    x = torch.cat((x, x, x), dim = 1)

    for name, layer in self.vgg_layers._modules.items():
      x = layer(x)
      if name == "relu2_2":
        break # stop forward at relu2_2
    return x

  def forward(self, inputs, targets):
    loss = 0
    if isinstance(inputs, list):
      for x, y in zip(inputs, targets):
        loss += F.mse_loss(self.vgg_feature(x), self.vgg_feature(y))
    else:
      loss = F.mse_loss(self.vgg_feature(inputs), self.vgg_feature(targets))
    return loss

class CSCP(nn.Module):
  """
    Implements Cross-Scale Consistency Pursuit Loss
  """
  def __init__(self):
    super(CSCP, self).__init__()

  def forward(self, density_maps, density_chunks):
    density_joined = torch.cat((torch.cat((density_chunks[0], density_chunks[1]), dim = 3), torch.cat((density_chunks[2], density_chunks[3]), dim = 3)), dim = 2)
    loss = F.mse_loss(density_maps, density_joined)
    return loss

class Adversarial(nn.Module):
  """
    Implements Adversarial Loss for ACSCP network
  """
  def __init__(self):
    super(Adversarial, self).__init__()

  def forward(self, real_l_logits, real_s_logits, fake_l_logits, fake_s_logits, mode = "generator"):
    assert mode in list(["discriminator", "generator"]), ValueError("Invalid network mode '{}'".format(mode))
    loss = 0
    ones = torch.ones_like(fake_l_logits)
    zeros = torch.zeros_like(fake_l_logits)
    if mode == "generator":
      # large subnet
      loss += F.binary_cross_entropy_with_logits(fake_l_logits, ones)
      # small subnet
      for fake_logit in fake_s_logits:
        loss += F.binary_cross_entropy_with_logits(fake_logit, ones)
    else:
      # large subnet
      loss += F.binary_cross_entropy_with_logits(fake_l_logits, zeros)
      loss += F.binary_cross_entropy_with_logits(real_l_logits, ones)
      # small subnet
      for fake_logit, real_logit in zip(fake_s_logits, real_s_logits):
        loss += F.binary_cross_entropy_with_logits(fake_logit, zeros)
        loss += F.binary_cross_entropy_with_logits(real_logit, ones)
    return loss

