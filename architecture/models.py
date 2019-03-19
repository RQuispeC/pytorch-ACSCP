from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['ACSCP']

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
		self.encoder_1 = Conv2d(1, 64, 6, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_2 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_3 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_4 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_5 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_6 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_7 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_8 = Conv2d(64, 64, 3, stride = 1, bn = True, activation = 'leakyrelu', dropout=False)
		
		self.decoder_1 = Deconv2d(64, 64, 3, stride = 1, bn = True, activation = 'relu', dropout = True)
		self.decoder_2 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = True)
		self.decoder_3 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = True)
		self.decoder_4 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = False)
		self.decoder_5 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = False)
		self.decoder_6 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = False)
		self.decoder_7 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = False)
		self.decoder_8 = Deconv2d(128, 1, 6, stride = 2, bn = True, activation = 'tanh', dropout = False)
	
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
		d = F.upsample(d, size = (e7.size()[-2], e7.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e7), dim=1)
		d = self.decoder_2(d)
		d = F.upsample(d, size = (e6.size()[-2], e6.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e6), dim=1)
		d = self.decoder_3(d)
		d = F.upsample(d, size = (e5.size()[-2], e5.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e5), dim=1)
		d = self.decoder_4(d)
		d = F.upsample(d, size = (e4.size()[-2], e4.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e4), dim=1)
		d = self.decoder_5(d)
		d = F.upsample(d, size = (e3.size()[-2], e3.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e3), dim=1)
		d = self.decoder_6(d)
		d = F.upsample(d, size = (e2.size()[-2], e2.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e2), dim=1)
		d = self.decoder_7(d)
		d = F.upsample(d, size = (e1.size()[-2], e1.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e1), dim=1)
		d = self.decoder_8(d)

		d = F.upsample(d, size = (x.size()[-2], x.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		return d


class G_Small(nn.Module):
	def __init__(self):
		super(G_Small, self).__init__()
		self.encoder_1 = Conv2d(1, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_2 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_3 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_4 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_5 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_6 = Conv2d(64, 64, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.encoder_7 = Conv2d(64, 64, 3, stride = 1, bn = True, activation = 'leakyrelu', dropout=False)
		
		self.decoder_1 = Deconv2d(64, 64, 3, stride = 1, bn = True, activation = 'relu', dropout = True)
		self.decoder_2 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = True)
		self.decoder_3 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = True)
		self.decoder_4 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = False)
		self.decoder_5 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = False)
		self.decoder_6 = Deconv2d(128, 64, 4, stride = 2, bn = True, activation = 'relu', dropout = False)
		self.decoder_7 = Deconv2d(128, 1, 4, stride = 2, bn = True, activation = 'relu', dropout = False)
	
	def forward(self, x):
		e1 = self.encoder_1(x)
		e2 = self.encoder_2(e1)
		e3 = self.encoder_3(e2)
		e4 = self.encoder_4(e3)
		e5 = self.encoder_5(e4)
		e6 = self.encoder_6(e5)
		e7 = self.encoder_7(e6)

		d = self.decoder_1(e7)
		d = F.upsample(d, size = (e6.size()[-2], e6.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e6), dim=1)
		d = self.decoder_2(d)
		d = F.upsample(d, size = (e5.size()[-2], e5.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e5), dim=1)
		d = self.decoder_3(d)
		d = F.upsample(d, size = (e4.size()[-2], e4.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e4), dim=1)
		d = self.decoder_4(d)
		d = F.upsample(d, size = (e3.size()[-2], e3.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e3), dim=1)
		d = self.decoder_5(d)
		d = F.upsample(d, size = (e2.size()[-2], e2.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e2), dim=1)
		d = self.decoder_6(d)
		d = F.upsample(d, size = (e1.size()[-2], e1.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		d = torch.cat((d, e1), dim=1)
		d = self.decoder_7(d)

		d = F.upsample(d, size = (x.size()[-2], x.size()[-1]), mode = 'bilinear') #fix odd shape tensors
		return d

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.f_1 = Conv2d(1, 48, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.f_2 = Conv2d(48,96, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.f_3 = Conv2d(96, 192, 4, stride = 2, bn = True, activation = 'leakyrelu', dropout=False)
		self.f_4 = Conv2d(192, 381, 3, stride = 1, bn = True, activation = 'leakyrelu', dropout=False)
		self.f_5 = Conv2d(381, 1, 3, stride = 1, bn = True, activation = 'leakyrelu', dropout=False)

	def forward(self, x):
		x = self.f_1(x)
		x = self.f_2(x)
		x = self.f_3(x)
		x = self.f_4(x)
		x = self.f_5(x)

		logits = F.avg_pool2d(x, x.size()[2:])
		logits = logits.view(logits.size(0), -1)
		y = F.tanh(logits)
		return logits, y

class ACSCP(nn.Module):
	def __init__(self):
		super(ACSCP, self).__init__()
		self.g_large = G_Large()
		self.g_small = G_Small()
		self.d_large = discriminator()
		self.d_small = discriminator()


	def forward(self, inputs, targets, mode = "generator"):
		assert mode in list(["discriminator", "generator"]), ValueError("Invalid network mode '{}'".format(mode))

		# forward g_large
		g_l = self.g_large(inputs)
		if not self.training:
			return g_l

		# create chunks
		chunks = torch.chunk(inputs, chunks = 2, dim = 2)
		inputs_1, inputs_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
		inputs_3, inputs_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)
		
		chunks = torch.chunk(targets, chunks = 2, dim = 2)
		targets_1, targets_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
		targets_3, targets_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

		inputs_chunks = list([inputs_1, inputs_2, inputs_3, inputs_4])
		targets_chunks = list([targets_1, targets_2, targets_3, targets_4])

		# forward g_small
		g_s = []
		for chunk in inputs_chunks:
			g_s.append(self.g_small(chunk))

		# forward d_large
		real_l_logits = None
		fake_l_logits = None
		if mode == "generator": # use only fake/generated maps
			fake_l_logits, _ = self.d_large(g_l)
		else: # use fake and grount thruth maps
			fake_l_logits, _ = self.d_large(g_l)
			real_l_logits, _ = self.d_large(inputs_d_l)

		# forward d_small
		real_s_logits = []
		fake_s_logits = []
		for real, fake in zip(targets_chunks, g_s):
			if mode == "generator": # use only fake/generated maps
				fake_s_logits_item, _ = self.d_small(fake)
				real_s_logits_item = None
			else: # use fake and grount thruth maps
				fake_s_logits_item, _ = self.d_small(fake)
				real_s_logits_item, _ = self.d_small(real)

			fake_s_logits.append(fake_s_logits_item)
			real_s_logits.append(real_s_logits_item)

		return g_l, g_s, targets, targets_chunks, real_l_logits, real_s_logits, fake_l_logits, fake_s_logits, density_joined
