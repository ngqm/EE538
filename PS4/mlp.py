import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(0)

def ReLU(X):
	try:
		return np.array([max(0,x) for x in X])
	except:
		return max(0,X)
def dReLU(X):
	try:
		return np.array([1 if x>0 else 0 for x in X])
	except:
		if X>0: 
			return 1
		else: 
			return 0

sigmoid = lambda X: (1 + np.exp(-X))**(-1)
dsigmoid = lambda X: sigmoid(X)*(1-sigmoid(X))

def unison_shuffles(a, b):
	p = np.random.permutation(len(a))
	return a[p], b[p]

class MLP:

	def __init__(self, width, activation=ReLU, n_layer=1):
		'''
		We only consider 1 output neuron
		'''
		self.activation = activation	# nonlinearity
		if activation == ReLU:
			self.dact = dReLU
		if activation == sigmoid:
			self.dact = dsigmoid

		self.width		= width				# array of number of neurons for each layer
		self.n_layer 	= n_layer			# number of hidden layers + 1

		self.alpha		= .02				# momentum rate
		self.eta0 		= .02				# learning rate

		self.y 			= [0]*(n_layer+1)
		self.v 			= [0]*(n_layer+1)
		self.w 			= [0]*(n_layer+1)
		for layer in range(1, self.n_layer+1):
			self.w[layer] = .1*np.random.randn(self.width[layer],
				self.width[layer-1]+1)

		self.dy 		= [0]*(n_layer+1)
		self.dw 		= [0]*(n_layer+1)
		for layer in range(1, self.n_layer+1):
			self.dw[layer] = np.random.randn(self.width[layer], 
				self.width[layer-1]+1)
			self.dy[layer] = np.random.randn(self.width[layer])

	def forward(self, point):

		self.y[0] = [point[0], point[1]]
		for layer in range(1, self.n_layer+1):
			self.v[layer] = np.dot(self.w[layer],np.append(1,self.y[layer-1]))
			self.y[layer] = self.activation(self.v[layer])

	def backward(self, point, d):

		d_dash = 0 if d == -1 else 1

		e = d_dash - self.y[-1][0]
		self.dy[-1] = [-e]
		self.dw[-1][0][0] = -self.dact(self.v[-1][0])*e
		for b in range(1, self.width[-2]+1):
			self.dw[-1][0][b] = -self.y[-2][b-1]*self.dact(self.v[-1][0])*e
		
		for layer in range(self.n_layer-1, 0, -1):
			for a in range(self.width[layer]):
				self.dy[layer][a] = sum([self.dy[layer+1][c]*self.dact(self.v[layer+1][c])*self.w[layer+1][c][a+1] for c in range(self.width[layer+1])])
				self.dw[layer][a][0] = -self.dact(self.v[layer][a])*self.dy[layer][a]
				for b in range(1, self.width[layer-1]+1):
					self.dw[layer][a][b] = -self.y[layer-1][b-1]*self.dact(self.v[layer][a])*self.dy[layer][a]



	def predict(self, point):

		y 			= [0]*(self.n_layer+1)
		v 			= [0]*(self.n_layer+1)
		w 			= [0]*(self.n_layer+1)
		for layer in range(1, self.n_layer+1):
			w[layer] = np.random.randn(self.width[layer],
				self.width[layer-1]+1)

		y[0] = [point[0], point[1]]
		for layer in range(1, self.n_layer+1):
			v[layer] = np.dot(self.w[layer],np.append(1,y[layer-1]))
			y[layer] = self.activation(v[layer])

		if y[-1][0] > .5:
			return 1
		else:
			return -1

	def accuracy(self, data, label):
		correct = sum([1 for x, y in zip(data, label) if self.predict(x) == y])
		return correct/len(data)

	def learn(self, data, label):

		self.log = []
		for epoch in range(3):
			print(self.w)
			eta = self.eta0/(1+epoch)
			epoch_data, epoch_label = unison_shuffles(data, label)
			for point, d in zip(epoch_data, epoch_label):
				self.forward(point)
				self.backward(point, d)
				for layer in range(1,self.n_layer+1):
					for a in range(self.width[layer]):
						for b in range(self.width[layer-1] + 1):
							self.w[layer][a][b] = self.alpha*self.w[layer][a][b] - eta*self.dw[layer][a][b]
			self.log.append(self.accuracy(data, label))


	def mlpcolor_code(self, data):
		"""
		Predict class for data and 
		return 2 lists
		"""
		train_data = data
		code1 = []
		code0 = []
		for x in train_data:
			if self.predict(x) == 1:
				code1.append(x)
			else:
				code0.append(x)
		code0, code1 = np.array(code0), np.array(code1)

		return np.array([code0, code1])