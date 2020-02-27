import numpy as np

from abc import abstractmethod

from activations import Sigmoid

class Layers:
	def forward(self):
		pass

	@abstractmethod
	def backward(self):
		pass

class Dense(Layers):
	def __init__(self,num_units, intial_weights, intial_biases):

		self.units=num_units
		self.w=intial_weights
		self.b=intial_biases


	def forward(self,x):
		self.inputs=x
		return np.matmul(x, self.w) #Normal Matrix Multiplication

	def gradient(self):
		return self.W.T

	def backward(self):
		return np.dot(global_gradient, self.gradient())

class Activation(Layers):
	activations={
		"sigmoid":Sigmoid
	}

	def __init__(self,name):
		self._fun=self.activations[name]()
		#self._func= Sigmoid()

	def forward(self,x):
		self.inputs=x
		return self._func.forward(x)# sigmoid.forward(x)

	def backward(self,global_gradient):
		return global_gradient*self._func.gradient(self.inputs)