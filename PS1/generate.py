import numpy as np 
import pandas as pd 


np.random.seed(0) # for reproduction

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


# P1

C1 = [[0,0], [3,0], [0,1.5]]
R1 = [1, 2, 1/2]
N1 = [100, 100, 100]

generate(type = 'uniform', C = C1, R = R1, 
	N = N1, path = 'P1/P1.csv')


# P2

Mean2 = [[0,0], [3,0], [0,1.5]]
Cov2 = [[[1, 0], [0, 1]],
	[[4, 0], [0, 4]],
	[[1/4, 0], [0, 1/4]]]
N2 = [100, 100, 100]

generate(type = 'gaussian', mean = Mean2, 
	cov = Cov2, N = N2, path = 'P2/P2.csv')

# P3

C3 = [[0,0], [3,0], [0,1.5]]
R3 = [1, 2, 1/2]
N3 = [100, 200, 100]

generate(type = 'uniform', C = C3, R = R3, 
	N = N3, path = 'P3/P3.csv')


# P4

Mean4 = [[0,0], [3,0], [0,1.5]]
Cov4 = [[[1, 0], [0, 1]],
	[[4, 0], [0, 4]],
	[[1/4, 0], [0, 1/4]]]
N4 = [100, 200, 100]

generate(type = 'gaussian', mean = Mean4, 
	cov = Cov4, N = N4, path = 'P4/P4.csv')