import torch
import torch.nn as nn
from torch.nn import functional as F
from architecture import network
from architecture.models import ACSCP
from architecture.losses import MSE, Perceptual, CSCP

import numpy as np

class CrowdCounter(nn.Module):
	def __init__(self):
		super(CrowdCounter, self).__init__()
		self.euclidean_loss = MSE()
		self.perceptual_loss = Perceptual()
		self.cscp_loss = CSCP()
		self.GAN = ACSCP()
		self.alpha_euclidean = 150.0
		self.alpha_perceptual = 150.0
		self.alpha_cscp = 0.0
		self.loss_gen_large = 0.0
		self.loss_gen_small = 0.0
		self.loss_dis_large = 0.0
		self.loss_dis_small = 0.0

	def adv_loss_generator_large(self, inputs, gt_data): #adversarial loss for g_large
		batch_size, _, _, _ = inputs.size()
		x = self.GAN.generator_large(inputs)
		fake_logits = self.GAN.discriminator_large(inputs, x)
		ones = torch.ones(batch_size)
		ones = ones.cuda()
		loss = F.binary_cross_entropy_with_logits(fake_logits, ones)
		return x, loss
	
	def adv_loss_generator_small(self, inputs_chunks, gt_data_chunks): #adversarial loss for g_small
		batch_size, _, _, _ = inputs_chunks[0].size()
		x = self.GAN.generator_small(inputs_chunks)
		fake_logits = self.GAN.discriminator_small(inputs_chunks, x)
		ones = torch.ones(batch_size)
		ones = ones.cuda()
		loss = 0
		for fake_logit in fake_logits:
			loss += F.binary_cross_entropy_with_logits(fake_logit, ones)
		return x, loss
	
	def adv_loss_discriminator_large(self, inputs, gt_data): #adversarial loss for d_large
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

		x = self.GAN.generator_large(inputs)
		fake_logits = self.GAN.discriminator_large(inputs, x)

		real_logits = self.GAN.discriminator_large(inputs, gt_data)

		loss_fake = F.binary_cross_entropy_with_logits(fake_logits, zeros)
		loss_real = F.binary_cross_entropy_with_logits(real_logits, ones)
		loss = loss_fake + loss_real
		return x, loss

	def adv_loss_discriminator_small(self, inputs_chunks, gt_data_chunks): #adversarial loss for d_small		
		batch_size, _, _, _ = inputs_chunks[0].size()
		ones = torch.ones(batch_size)
		# swap some labels and smooth the labels
		idx = np.random.uniform(0, 1, batch_size)
		idx = np.argwhere(idx < 0.03).reshape(-1)
		ones += torch.tensor(np.random.uniform(-0.1, 0.1))
		ones[idx] = 0
		zeros = torch.zeros(batch_size)
		ones = ones.cuda()
		zeros = zeros.cuda()

		x = self.GAN.generator_small(inputs_chunks)
		fake_logits = self.GAN.discriminator_small(inputs_chunks, x)
		real_logits = self.GAN.discriminator_small(inputs_chunks, gt_data_chunks)
		
		loss_fake = 0
		loss_real = 0
		for fake_logit, real_logit in zip(fake_logits, real_logits):
			loss_fake += F.binary_cross_entropy_with_logits(fake_logit, zeros)
			loss_real += F.binary_cross_entropy_with_logits(real_logit, ones)
		loss = loss_fake + loss_real
		return x, loss
	
	def chunk_input(self, inputs, gt_data):
		chunks = torch.chunk(inputs, chunks = 2, dim = 2)
		inputs_1, inputs_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
		inputs_3, inputs_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

		chunks = torch.chunk(gt_data, chunks = 2, dim = 2)
		targets_1, targets_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
		targets_3, targets_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

		inputs_chunks = list([inputs_1, inputs_2, inputs_3, inputs_4])
		targets_chunks = list([targets_1, targets_2, targets_3, targets_4])

		return inputs_chunks, targets_chunks

	def forward(self, inputs, gt_data=None, epoch=0, mode="generator"):
		assert mode in list(["discriminator", "generator"]), ValueError("Invalid network mode '{}'".format(mode))
		inputs = network.np_to_variable(inputs, is_cuda=True, is_training=self.training)
		if not self.training:
			g_l = self.GAN.generator_large(inputs)
		else:
			gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
			#chunk input data in 4
			inputs_chunks, gt_data_chunks = self.chunk_input(inputs, gt_data)
			if mode == "generator":
				# g_large
				x_l, self.loss_gen_large = self.adv_loss_generator_large(inputs, gt_data)
				self.loss_gen_large += self.alpha_euclidean * self.euclidean_loss(x_l, gt_data)
				self.loss_gen_large += self.alpha_perceptual * self.perceptual_loss(x_l, gt_data)

				# g_small
				x_s, self.loss_gen_small = self.adv_loss_generator_small(inputs_chunks, gt_data_chunks)
				self.loss_gen_small += self.alpha_euclidean * self.euclidean_loss(x_s, gt_data_chunks)
				self.loss_gen_small += self.alpha_perceptual * self.perceptual_loss(x_s, gt_data_chunks)

				if epoch >= 100:
					self.alpha_cscp = 10
				self.loss_gen_large += self.alpha_cscp * self.cscp_loss(x_l, x_s)
				self.loss_gen_small += self.alpha_cscp * self.cscp_loss(x_l, x_s)

				self.loss_gen = self.loss_gen_large + self.loss_gen_small
			else:
				#d_large
				x_l, self.loss_dis_large = self.adv_loss_discriminator_large(inputs, gt_data)

				#d_small
				x_s, self.loss_dis_small = self.adv_loss_discriminator_small(inputs_chunks, gt_data_chunks)
			g_l = x_l
		return g_l
