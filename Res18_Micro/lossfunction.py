import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torch.nn import Parameter
from torch.autograd import Variable

class FocalLoss(nn.Module):

	def __init__(self, num_classes, alpha=None, gamma=2):
		super(FocalLoss, self).__init__()
		self.num_classes = num_classes
		if alpha is None:
			self.alpha = torch.ones(self.num_classes, 1)
		else:
			self.alpha = torch.tensor(alpha)
		self.gamma = gamma
		self.sm = nn.Softmax(dim=1)

	def forward(self, inputs, labels):
		prob = self.sm(inputs)
		one_hot = torch.zeros(inputs.size())
		if inputs.is_cuda:
			one_hot = one_hot.cuda()
		one_hot.scatter_(1, labels.view(-1, 1).data, 1)
		prob = (prob * one_hot).sum(1).view(-1, 1)
		ce = prob.log()

		loss = -self.alpha * (torch.pow((1 - prob), self.gamma)) * ce
		return loss.mean()

class Arcface(nn.Module):

	def __init__(self, in_features, num_classes, s=30.0, m=0.50, easy_margin=False):
		super(Arcface, self).__init__()
		self.in_features = in_features
		self.out_features = num_classes
		self.s = s
		self.m = m
		self.cosm = math.cos(self.m)
		self.sinm = math.sin(self.m)
		self.easy_margin = easy_margin
		self.weight = Parameter(torch.FloatTensor(self.out_features, self.in_features))
		nn.init.xavier_uniform_(self.weight)

	def forward(self, inputs, labels):
		cosine = Functional.linear(Functional.normalize(inputs), Functional.normalize(self.weight))
		sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
		phi = cosine * self.cosm - sine * self.sinm
		if self.easy_margin:
			phi = torch.where(cosine>0, phi, cosine)
		else:
			phi = torch.where(cosine+self.cosm>0, phi, cosine-self.sinm*self.m)
		one_hot = torch.zeros(cosine.size())
		if inputs.is_cuda:
			one_hot = one_hot.cuda()
		one_hot.scatter_(1, labels.view(-1, 1).data, 1)
		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
		output *= self.s

		return output
