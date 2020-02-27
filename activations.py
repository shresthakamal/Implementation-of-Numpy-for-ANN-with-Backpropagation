import numpy as np

from abc import abstractmethod


class Activations:
	@abstractmethod
	def forward(self,x):
		pass

	@abstractmethod
	def gradient(self,x):
		pass

class Sigmoid(Activations):
	def forward(self,x):
		return 1/(1+np.exp(-x))

	def gradient(self,x):
		return self.forward(x)* (1-self.forward(x))

