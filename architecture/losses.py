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
    loss = self.mse(inputs, targets)
    return loss

class Perceptual(nn.Module):
  """
        Computes Perceptual loss between inputs
  """
  def __init__(self):
    super(Perceptual, self).__init__()
    vgg = torchvision.models.vgg.vgg16(pretrained=True)
    self.vgg_layers = nn.Sequential(*list(vgg.features.children())[:9]) #recover up to relu2_2 layer

  def vgg_feature(self, x):
    # x has only one channel and vgg expects 3
    x = torch.cat((x, x, x), dim = 1)
    x = self.vgg_layers(x)
    return x

  def forward(self, inputs, targets):
    loss = F.mse_loss(self.vgg_feature(inputs), self.vgg_feature(targets))
    return loss

class CSCP(nn.Module):
  """
    Implements Cross-Scale Consistency Pursuit Loss
  """
  def __init__(self):
    super(CSCP, self).__init__()

  def forward(self, density_maps, density_chunks):
    batch_size, _, _, _ = density_maps.size()
    inputs_1 = density_chunks[:batch_size, :, :, :]
    inputs_2 = density_chunks[batch_size:2*batch_size, :, :, :]
    inputs_3 = density_chunks[2*batch_size:3*batch_size, :, :, :]
    inputs_4 = density_chunks[3*batch_size:4*batch_size, :, :, :]
    density_joined = torch.cat((torch.cat((inputs_1, inputs_2), dim = 3), torch.cat((inputs_3, inputs_4), dim = 3)), dim = 2)
    loss = F.mse_loss(density_maps, density_joined)
    return loss

