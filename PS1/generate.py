import numpy as np 
import pandas as pd 


np.random.seed(0) # for reproduction


# CONVENIENT FUNCTIONS


def rand(a):
	"""
	Return uniform random number 
	between -a and a, where a > 0
	"""
	return 2*a*np.random.rand() - a


def generate(type, **param):
	"""
	Return csv file at path with 
	3 clusters following distribution type
	type = 'uniform':
	**param: C, R, N, path
	type = 'gaussian':
	**param: mean, cov, N, path
	"""
	if type == 'uniform':
		C = param.get('C')
		R = param.get('R')
		N = param.get('N')
		path = param.get('path')

		clusters = [[],[],[]]

		for i in range(3):
			for _ in range(N[i]):
				x = C[i][0] + rand(R[i])
				y = C[i][1] + rand(np.sqrt(R[i]**2 - (x - C[i][0])**2))
				clusters[i].append(np.array([x, y]))
		
		pd.DataFrame(clusters).to_csv(path, 
			header = None, index = None)

	if type == 'gaussian':
		mean = param.get('mean')
		cov = param.get('cov')
		N = param.get('N')
		path = param.get('path')

		clusters = [[],[],[]]
		
		for i in range(3):
			for _ in range(N[i]):
				x, y = np.random.multivariate_normal(mean[i], cov[i])
				clusters[i].append(np.array([x, y]))
		
		pd.DataFrame(clusters).to_csv(path, 
			header = None, index = None)


# DATA GENERATION


C = [[[0,0], [2,0], [0,2]],		# P1
	[[0,0], [3,0], [0, 1.5]],	# P2
	[[0,0], [2,0], [0,2]],		# P3
	[[0,0], [3,0], [0, 1.5]]]	# P4

R = [[1, 1, 1],		# P1
	[1, 2, .5],		# P2
	[1, 1, 1],		# P3
	[1, 2, .5]]		# P4

N = [[100, 100, 100],	# P1
	[100, 100, 100],	# P2
	[100, 200, 100],	# P3
	[100, 200, 100]]	# P4

for problem in range(4):

	p = str(problem + 1)

	generate(type = 'uniform', C = C[problem],
		R = R[problem], N = N[problem],
		path = 'P' + p + '/' + 'P' + p + '.csv')