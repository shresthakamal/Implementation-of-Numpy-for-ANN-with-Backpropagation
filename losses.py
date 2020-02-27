import numpy as np

from abc import abstractmethod

class Loss:
	@abstractmethod
	def loss(self,y_true,y_pred):
		pass
	def gradient(self,y_true,y_pred):
		pass

class MSE:
	def loss(self,y_true,y_pred):
		return 0.5* np.power((y_true-y_pred),2) #Squared Error

	def gradient(self,y_true,y_pred):
		return -1* (y_true-y_pred)