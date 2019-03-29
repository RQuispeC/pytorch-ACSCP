import torch
import torch.nn as nn
from torch.nn import functional as F
from architecture import network
from architecture.models import G_Large, G_Small, discriminator
from architecture.losses import MSE, Perceptual, CSCP

import numpy as np

class CrowdCounter(nn.Module):
	def __init__(self):
		super(CrowdCounter, self).__init__()
		self.euclidean_loss = MSE()
		self.perceptual_loss = Perceptual()
		self.cscp_loss = CSCP()
		self.g_large = G_Large()
		self.g_small = G_Small()
		self.d_large = discriminator()
		self.d_small = discriminator()
		self.alpha_euclidean = 150.0
		self.alpha_perceptual = 150.0
		self.alpha_cscp = 0.0
		self.loss_gen_large = 0.0
		self.loss_gen_small = 0.0
		self.loss_dis_large = 0.0
		self.loss_dis_small = 0.0

	def adv_loss_generator(self, generator, discriminator, inputs):
		batch_size, _, _, _ = inputs.size()
		x = generator(inputs)
		fake_logits, _ = discriminator(inputs, x)
		ones = torch.ones(batch_size).cuda()
		loss = F.binary_cross_entropy_with_logits(fake_logits, ones)
		return x, loss
	
	def adv_loss_discriminator(self, generator, discriminator, inputs, targets):
		batch_size, _, _, _ = inputs.size()
		ones = torch.ones(batch_size)
		# swap some labels and smooth the labels
		idx = np.random.uniform(0, 1, batch_size)
		idx = np.argwhere(idx < 0.03).reshape(-1)
		ones += torch.tensor(np.random.uniform(-0.1, 0.1))
		ones[idx] = 0
		zeros = torch.zeros(batch_size)
		ones = ones.cuda()
		zeros = zeros.cuda()

		x = generator(inputs)
		fake_logits, _ = discriminator(inputs, x)
		real_logits, _ = discriminator(inputs, targets)

		loss_fake = F.binary_cross_entropy_with_logits(fake_logits, zeros)
		loss_real = F.binary_cross_entropy_with_logits(real_logits, ones)
		loss = loss_fake + loss_real
		return x, loss
	
	def chunk_input(self, inputs, gt_data):
		chunks = torch.chunk(inputs, chunks = 2, dim = 2)
		inputs_1, inputs_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
		inputs_3, inputs_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

		chunks = torch.chunk(gt_data, chunks = 2, dim = 2)
		targets_1, targets_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
		targets_3, targets_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

		inputs_chunks = torch.cat((inputs_1, inputs_2, inputs_3, inputs_4), dim = 0)
		targets_chunks = torch.cat((targets_1, targets_2, targets_3, targets_4), dim = 0)

		return inputs_chunks, targets_chunks

	def forward(self, inputs, gt_data=None, epoch=0, mode="generator"):
		assert mode in list(["discriminator", "generator"]), ValueError("Invalid network mode '{}'".format(mode))
		inputs = network.np_to_variable(inputs, is_cuda=True, is_training=self.training)
		if not self.training:
			g_l = self.g_large(inputs)
		else:
			gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
			#chunk input data in 4
			inputs_chunks, gt_data_chunks = self.chunk_input(inputs, gt_data)

			if mode == "generator":
				# g_large
				x_l, self.loss_gen_large = self.adv_loss_generator(self.g_large, self.d_large, inputs)
				self.loss_gen_large += self.alpha_euclidean * self.euclidean_loss(x_l, gt_data)
				self.loss_gen_large += self.alpha_perceptual * self.perceptual_loss(x_l, gt_data)

				# g_small
				x_s, self.loss_gen_small = self.adv_loss_generator(self.g_small, self.d_small, inputs_chunks)
				self.loss_gen_small += self.alpha_euclidean * self.euclidean_loss(x_s, gt_data_chunks)
				self.loss_gen_small += self.alpha_perceptual * self.perceptual_loss(x_s, gt_data_chunks)

				if epoch >= 100:
					self.alpha_cscp = 10
				self.loss_gen_large += self.alpha_cscp * self.cscp_loss(x_l, x_s)
				self.loss_gen_small += self.alpha_cscp * self.cscp_loss(x_l, x_s)

				self.loss_gen = self.loss_gen_large + self.loss_gen_small
			else:
				#d_large
				x_l, self.loss_dis_large = self.adv_loss_discriminator(self.g_large, self.d_large, inputs, gt_data)

				#d_small
				x_s, self.loss_dis_small = self.adv_loss_discriminator(self.g_small, self.d_small, inputs_chunks, gt_data_chunks)
			g_l = x_l
		return g_l
