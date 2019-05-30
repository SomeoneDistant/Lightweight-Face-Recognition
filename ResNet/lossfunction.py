import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torch.nn import Parameter
from torch.autograd import Variable

class FocalLoss(nn.Module):

	def __init__(self, alpha=1, gamma=2):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.ce = nn.CrossEntropyLoss()

	def forward(self, inputs, labels):
		logp = self.ce(inputs, labels)
		prob = torch.exp(-logp)
		loss = (self.alpha * (torch.pow((1 - prob), self.gamma)) * logp).mean()
		return loss

class Arcface(nn.Module):

	def __init__(self, features, num_classes, s=64.0, m=0.50):
		super(Arcface, self).__init__()
		self.features = features
		self.classes = num_classes
		self.weight = Parameter(torch.FloatTensor(num_classes, features))
		nn.init.xavier_uniform_(self.weight)

		self.s = s
		self.m = m
		self.cosm = math.cos(m)
		self.sinm = math.sin(m)

	def forward(self, inputs, labels):
		cosine = Functional.linear(Functional.normalize(inputs), Functional.normalize(self.weight))
		sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
		phi = cosine * self.cosm - sine * self.sinm
		phi = torch.where(cosine+self.cosm>0, phi, cosine-self.sinm*self.m)
		one_hot = torch.zeros(cosine.size())
		if inputs.is_cuda:
			one_hot = one_hot.cuda()
		one_hot.scatter_(1, labels.view(-1, 1).data, 1)
		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
		output *= self.s

		return output, cosine
