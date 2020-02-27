import numpy as np


from layers import Dense, Activation
from activations import Sigmoid
from losses import MSE

if __name__=="__main__":
	x=np.array([[0.05, .1]])
	W1=np.array([
		[.15,.20],
		[.25,.30]
	])

	W2=np.array([
		[.40,.45],
		[.50,.55]
		])

	b1=.35
	b2=0.60

	y_true=np.array([[.01, .99]])


	#Layers Generation 
	dense=Dense(2,W1,b1)
	dense2=Dense(2,W2,b2)

	activation1=Sigmoid()
	# activation2=Sigmoid()
	activation2=Activation("sigmoid")

	loss_func=MSE()

	#Forward Pass
	# Dense -> Activation -> Dense -> Activation -> y_pred

	z1= dense.forward(x)
	a1=activation1.forward(z1)
	print("Activation Value:",a1)

	z2=dense2.forward(a1)
	a2=activation2.forward(z2)
	y_pred=a2


	loss=loss_func.loss(y_true,y_pred)

	print("Individual Loss:",loss)
	total_loss=np.mean(loss)
	print("Total Loss:",total_loss)


	#Backward Propagation
	dLdy_pred=loss_func.gradient(y_true,y_pred)
	print("dLdy:", dLdy_pred)

	'''

	dydz=activation2.gradient(z2)
	dLdz2-dLdy_pred*dydz
	
	'''

	
	dLdz2=activation2.backward(dLdy_pred)
	dLda1=dense2.backward(dLdz2)
	dLdz1=sigmoid.backward(dLda1)
	dLdw=dense.backward(dLdz1)

	




