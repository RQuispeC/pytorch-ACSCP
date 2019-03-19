import torch.nn as nn
from architecture import network
from architecture.models import ACSCP
from architecture.losses import Adversarial, MSE, Perceptual, CSCP

class CrowdCounter(nn.Module):
	def __init__(self):
		super(CrowdCounter, self).__init__()
		self.adversarial_loss = Adversarial()
		self.euclidean_loss = MSE()
		self.perceptual_loss = Perceptual()
		self.cscp_loss = CSCP()
		self.net = ACSCP()
		self.alpha_euclidean = 150.0
		self.alpha_perceptual = 150.0
		self.alpha_cscp = 0.0
		self.loss = 0.0

	def forward(self, inputs, gt_data=None, epoch=0, mode="generator"):
		self.loss = 0.0
		inputs = network.np_to_variable(inputs, is_cuda=True, is_training=self.training)
		gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
		if not self.training:
			g_l = self.net(inputs, gt_data)
		else:
			g_l, g_s, targets_l, targets_s, real_l_logits, real_s_logits, fake_l_logits, fake_s_logits = self.net(inputs, gt_data, mode=mode)
			if epoch >= 100:
				self.alpha_cscp = 10
			
			self.loss += self.adversarial_loss(real_l_logits, real_s_logits, fake_l_logits, fake_s_logits, mode=mode)
			self.loss += self.alpha_euclidean * (self.euclidean_loss(g_l, targets_l) + self.euclidean_loss(g_s, targets_s))
			self.loss += self.alpha_perceptual * (self.perceptual_loss(g_l, targets_l) + self.perceptual_loss(g_s, targets_s))
			self.loss += self.alpha_cscp * (self.cscp_loss(g_l, g_s))

		return g_l
