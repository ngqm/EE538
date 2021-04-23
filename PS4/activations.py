import numpy as np

# activation functions and their derivatives

def sigmoid(x):
	return (np.tanh(x/2)+1)/2

def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))

def ReLU(x):
	new_x = np.array(x)
	return new_x * (new_x > 0)

def ReLU_prime(x):
	new_x = np.array(x)
	return np.ones(new_x.shape)*(new_x > 0)