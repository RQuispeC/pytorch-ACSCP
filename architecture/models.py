from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F

class Conv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=False, activation = 'leakyrelu', dropout = False):
		super(Conv2d, self).__init__()
		padding = int((kernel_size - 1) / 2)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
		self.dropout = nn.Dropout(p=0.5) if dropout else None
		if activation == 'leakyrelu':
			self.activation = nn.LeakyReLU(negative_slope = 0.2)
		elif activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'tanh':
			self.activation = nn.Tanh()
		else:
			raise ValueError('Not a valid activation, received {}'.format(activation))

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = self.activation(x)
		return x

class Deconv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=False, activation = 'leakyrelu', dropout = False):
		super(Deconv2d, self).__init__()
		padding = int((kernel_size - 1) / 2)
		self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
		self.dropout = nn.Dropout(p=0.5) if dropout else None
		if activation == 'leakyrelu':
			self.activation = nn.LeakyReLU(negative_slope = 0.2)
		elif activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'tanh':
			self.activation = nn.Tanh()
		else:
			raise ValueError('Not a valid activation, received {}'.format(activation))

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = self.activation(x)
		return x

class G_Large(nn.Module):
	def __init__(self):
		super(G_Large, self).__init__()
		self.encoder_1 = Conv2d(1, 64, 6, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_2 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_3 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_4 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_5 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_6 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_7 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_8 = Conv2d(64, 64, 3, stride = 1, bn = False, activation = 'leakyrelu', dropout=False)
		
		self.decoder_1 = Deconv2d(64, 64, 3, stride = 1, bn = False, activation = 'relu', dropout = True)
		self.decoder_2 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = True)
		self.decoder_3 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = True)
		self.decoder_4 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = False)
		self.decoder_5 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = False)
		self.decoder_6 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = False)
		self.decoder_7 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = False)
		self.decoder_8 = Deconv2d(128, 1, 6, stride = 2, bn = False, activation = 'relu', dropout = False)
	
	def forward(self, x):
		e1 = self.encoder_1(x)
		e2 = self.encoder_2(e1)
		e3 = self.encoder_3(e2)
		e4 = self.encoder_4(e3)
		e5 = self.encoder_5(e4)
		e6 = self.encoder_6(e5)
		e7 = self.encoder_7(e6)
		e8 = self.encoder_8(e7)

		d = self.decoder_1(e8)
		d = torch.cat((d, e7), dim=1)
		d = self.decoder_2(d)
		d = torch.cat((d, e6), dim=1)
		d = self.decoder_3(d)
		d = torch.cat((d, e5), dim=1)
		d = self.decoder_4(d)
		d = torch.cat((d, e4), dim=1)
		d = self.decoder_5(d)
		d = torch.cat((d, e3), dim=1)
		d = self.decoder_6(d)
		d = torch.cat((d, e2), dim=1)
		d = self.decoder_7(d)
		d = torch.cat((d, e1), dim=1)
		d = self.decoder_8(d)

		return d


class G_Small(nn.Module):
	def __init__(self):
		super(G_Small, self).__init__()
		self.encoder_1 = Conv2d(1, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_2 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_3 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_4 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_5 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_6 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.encoder_7 = Conv2d(64, 64, 3, stride = 1, bn = False, activation = 'leakyrelu', dropout=False)
		
		self.decoder_1 = Deconv2d(64, 64, 3, stride = 1, bn = False, activation = 'relu', dropout = True)
		self.decoder_2 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = True)
		self.decoder_3 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = True)
		self.decoder_4 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = False)
		self.decoder_5 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = False)
		self.decoder_6 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = False)
		self.decoder_7 = Deconv2d(128, 1, 4, stride = 2, bn = False, activation = 'relu', dropout = False)
	
	def forward(self, x):
		e1 = self.encoder_1(x)
		e2 = self.encoder_2(e1)
		e3 = self.encoder_3(e2)
		e4 = self.encoder_4(e3)
		e5 = self.encoder_5(e4)
		e6 = self.encoder_6(e5)
		e7 = self.encoder_7(e6)

		d = self.decoder_1(e7)
		d = torch.cat((d, e6), dim=1)
		d = self.decoder_2(d)
		d = torch.cat((d, e5), dim=1)
		d = self.decoder_3(d)
		d = torch.cat((d, e4), dim=1)
		d = self.decoder_4(d)
		d = torch.cat((d, e3), dim=1)
		d = self.decoder_5(d)
		d = torch.cat((d, e2), dim=1)
		d = self.decoder_6(d)
		d = torch.cat((d, e1), dim=1)
		d = self.decoder_7(d)

		return d

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.f_1 = Conv2d(2, 48, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.f_2 = Conv2d(48,96, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.f_3 = Conv2d(96, 192, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
		self.f_4 = Conv2d(192, 384, 3, stride = 1, bn = False, activation = 'leakyrelu', dropout=False)
		self.f_5 = Conv2d(384, 1, 3, stride = 1, bn = False, activation = 'leakyrelu', dropout=False)

	def forward(self, images, den_maps):
		x = torch.cat((images, den_maps), dim = 1)
		x = self.f_1(x)
		x = self.f_2(x)
		x = self.f_3(x)
		x = self.f_4(x)
		x = self.f_5(x)

		logits = F.avg_pool2d(x, x.size()[2:])
		logits = logits.view(-1)
		y = F.tanh(logits)
		return logits, y

