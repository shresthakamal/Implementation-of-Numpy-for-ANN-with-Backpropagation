import numpy as np

from abc import abstractmethod

class Layers:
	def forward(self):
		pass

	@abstractmethod
	def backward(self):
		pass

class Dense(Layers):
	def __init__(self,num_units):
		self.units=num_units
		self.w=
		self.b=