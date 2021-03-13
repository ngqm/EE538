import numpy as np 
import pandas as pd 


np.random.seed(0) # for reproduction

def rand(a):
	"""
	Return uniform random number 
	between -a and a, where a > 0
	"""
	return 2*a*np.random.rand() - a


# P1

C1 = [[0,0], [3,0], [0,1.5]]
R1 = [1, 2, 1/2]
N1 = [100, 100, 100]

clusters = [[],[],[]]

for i in range(3):
	for _ in range(N1[i]):
		x = C1[i][0] + rand(R1[i])
		y = C1[i][1] + rand(np.sqrt(R1[i]**2 - (x - C1[i][0])**2))
		clusters[i].append(np.array([x, y]))

pd.DataFrame(clusters).to_csv("P1/P1.csv", 
	header = None, index = None)


# P2


# P3

C3 = [[0,0], [3,0], [0,1.5]]
R3 = [1, 2, 1/2]
N3 = [100, 200, 100]

clusters = [[],[],[]]

for i in range(3):
	for _ in range(N3[i]):
		x = C3[i][0] + rand(R3[i])
		y = C3[i][1] + rand(np.sqrt(R3[i]**2 - (x - C3[i][0])**2))
		clusters[i].append(np.array([x, y]))

pd.DataFrame(clusters).to_csv("P3/P3.csv", 
	header = None, index = None)


# P4