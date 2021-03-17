import numpy as np 
import pandas as pd 


np.random.seed(0) # for reproduction

def csv_to_list(path):
	"""
	Turn data csv into separate 
	lists for x and y coordinates
	"""
	data = pd.read_csv(path, header = None)
	# 4 subsets for X and Y since the maximum
	# number of cluster is 4
	X, Y = [[], [], [], []], [[], [], [], []]
	for cluster in range(data.shape[0]):
		for point in range(data.shape[1]):
			try:
				x, y = data.iloc[cluster, point][1:-1].split()
				X[cluster].append(float(x))
				Y[cluster].append(float(y))
			except:
				pass

	return X, Y


def csv_to_numpy(path):
	"""
	Turn error/weight csv into a numpy array 
	"""
	array = pd.read_csv(path, header = None).to_numpy()
	return array


class SOM:

	def __init__(self, n_cluster, cooperative = True):

		self.n_cluster = n_cluster
		
		# topological neighborhood h
		sigma = 0.3 # effective width
		if cooperative:
			if n_cluster == 2:
				# squared lateral distance
				d2 = np.array([[0, 1],[1, 0]])
			if n_cluster == 3:
				d2 = np.array([[0, 3/4, 3/4],
							[3/4, 0, 3/4],
							[3/4, 3/4,0]])
			if n_cluster == 4:
				d2 = np.array([[0, 1, np.sqrt(2), 1],
							[1, 0, 1, np.sqrt(2)],
							[np.sqrt(2), 1, 0, 1],
							[1, np.sqrt(2), 1, 0]])
			self.h = np.exp(-d2/(2*sigma**2))
		else:
			self.h = np.identity(n_cluster)
		
		self.eta = 0.1 # learning rate
		self.weights = []
		self.error = []


	def learn(self, path, reuse_weights = False):
		"""
		Self-organize with data from path
		"""

		X, Y = csv_to_list(path)
		X_flat = X[0] + X[1] + X[2]
		Y_flat = Y[0] + Y[1] + Y[2]
		data = np.array([X_flat, Y_flat]).T

		# initialize weights from data
		if not reuse_weights:
			choices = np.random.choice(len(X_flat), self.n_cluster)
			for choice in choices:
				self.weights.append(data[choice])

		# compute error
		self.error.append(self.variance(data))


		# ITERATIVE LEARNING

		for epoch in range(1, 101):
			
			self.clusters = []
			# competition
			for i in range(self.n_cluster):
				self.clusters.append([x for x in data if self.winner(x)[0] == i])
				if len(self.clusters[-1]) != 0:
					mean = np.mean(self.clusters[-1], axis = 0)
					# adaptation & cooperation
					for j in range(self.n_cluster):
						self.weights[j] += self.eta*self.h[i,j]*(mean - self.weights[j])

			# compute error
			self.error.append(self.variance(data))

			if epoch < 100:
				print("Epoch {}".format(epoch), end="\r")
			else:
				print('Learning finished!')


	def winner(self, point):
		"""
		Return the winner given a input
		and the distance between them
		"""
		norm = np.linalg.norm(point - self.weights, axis = 1)
		winner = np.argmin(norm)
		
		return winner, norm[winner]


	def variance(self, data):
		"""
		Return within-cluster variance
		"""
		sum_error = 0
		for point in data:
			sum_error += self.winner(point)[1]

		return sum_error