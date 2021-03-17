import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
plt.style.use('seaborn')

from som import csv_to_list, csv_to_numpy
from generate import *

palette = list(sns.color_palette())


# CONVENIENT PLOT FUNCTIONS

margin = 0.5
def plot_cluster(path, xlim, ylim):
	"""
	Scatter plot from csv. xlim
	and ylim are manually set
	"""
	plt.xlim(xlim[0]-margin, xlim[1]+margin)
	plt.ylim(ylim[0]-margin, ylim[1]+margin)

	X, Y = csv_to_list(path)
	for cluster in range(len(X)):
		plt.scatter(X[cluster], Y[cluster], s = 10, alpha = .4)

def plot_error(path):
	"""
	Plot error curve from csv
	"""
	E = csv_to_numpy(path)
	X = range(len(E))
	plt.plot(X, E)
	plt.xlabel('Epoch')


def plot_weight(path):
	"""
	Scatter plot weight from csv
	"""
	W = csv_to_numpy(path)
	i = 0
	for w in W:
		plt.scatter(w[0], w[1], color = palette[i], marker = '^',
			s = 70)
		i += 1


# PLOTS


XLIM = [[-1,3], [-1,5], [-1,3], [-1,5]]
YLIM = [[-1,3], [-3, 3], [-1, 3], [-3, 3]]

title = ['P1: [$x^2 + y^2 < 1$], [$(x-2)^2 + y^2 < 1$], [$x^2 + (y-2)^2 < 1$], $N_1 = N_2 = N_3 = 100$', 
'P2: [$x^2 + y^2 < 1$], [$(x-3)^2 + y^2 < 4$], [$x^2 + (y-1.5)^2 < 0.25$], $N_1 = N_2 = N_3 = 100$', 
'P3: [$x^2 + y^2 < 1$], [$(x-2)^2 + y^2 < 1$], [$x^2 + (y-2)^2 < 1$], $N_1 = N_3 = 100$, $N_2 = 200$', 
'P4: [$x^2 + y^2 < 1$], [$(x-3)^2 + y^2 < 4$], [$x^2 + (y-1.5)^2 < 0.25$], $N_1 = N_3 = 100$, $N_2 = 200$']

for problem in range(4):

	p = str(problem + 1)

	plt.figure(figsize = (16,8))
	plt.suptitle(title[problem])

	plt.subplot(241)
	plt.title('(a) Generated data')
	plot_cluster('P' + p + '/' + 'P' + 
		p + '.csv', XLIM[problem], YLIM[problem])

	i = 0
	for c in C[problem]:
		plt.scatter(c[0], c[1], color = palette[i], marker = '^')
		i += 1

	n_cluster = [3, 2, 4]
	subproblem = ['b', 'c', 'd']	

	for i, j, k in zip(n_cluster, subproblem, range(2,5)):

		plt.subplot(240+k)
		plt.title('(' + j + ') ' + str(i) + ' learned clusters')
		plot_cluster('P' + p + '/' + j + '.csv', XLIM[problem], YLIM[problem])
		plot_weight('P' + p + '/' + j + 'W.csv')

		plt.subplot(244+k)
		plt.title('(' + j + ')  Loss function')
		plot_error('P' + p + '/' + j + 'E.csv')

	plt.savefig('P' + p + '/' + 'P' + p + '.jpg')


'''
# P2

plt.figure(figsize = (16,8))
plt.suptitle('P2: Gaussian clusters, $N_1 = N_2 = N_3 = 100$')

# Generated data
plt.subplot(241)
plt.title('(a) Generated data')
plot_cluster('P2/P2.csv', [-3,7], [-5,5])
i = 0
for m in Mean2:
	plt.scatter(m[0], m[1], color = palette[i], marker = '^')
	i += 1

#-----#

# 3 clusters
plt.subplot(242)
plt.title('(b) 3 learned clusters')
plot_cluster('P2/b.csv', [-3,7], [-5,5])
plot_weight('P2/bW.csv')

plt.subplot(246)
plt.title('(b) Learning curve')
plot_error('P2/bE.csv')

#-----#

# 2 clusters
plt.subplot(243)
plt.title('(c) 2 learned clusters')
plot_cluster('P2/c.csv', [-3,7], [-5,5])
plot_weight('P2/cW.csv')

plt.subplot(247)
plt.title('(c) Learning curve')
plot_error('P2/cE.csv')

#-----#

# 4 clusters
plt.subplot(244)
plt.title('(d) 4 learned clusters')
plot_cluster('P2/d.csv', [-3,7], [-5,5])
plot_weight('P2/dW.csv')

plt.subplot(248)
plt.title('(d) Learning curve')
plot_error('P2/dE.csv')

plt.savefig('P2/P2.jpg')


#----------#

# P3

plt.figure(figsize = (16,8))
plt.suptitle('P3: Uniform clusters, $N_1 = N_3 = 100$, $N_2 = 200$')

# Generated data
plt.subplot(241)
plt.title('(a) Generated data')
plot_cluster('P3/P3.csv', [-1,5], [-3,3])
i = 0
for c in C3:
	plt.scatter(c[0], c[1], color = palette[i], marker = '^')
	i += 1

#-----#

# 3 clusters
plt.subplot(242)
plt.title('(b) 3 learned clusters')
plot_cluster('P3/b.csv', [-1,5], [-3,3])
plot_weight('P3/bW.csv')

plt.subplot(246)
plt.title('(b) Learning curve')
plot_error('P3/bE.csv')

#-----#

# 2 clusters
plt.subplot(243)
plt.title('(c) 2 learned clusters')
plot_cluster('P3/c.csv', [-1,5], [-3,3])
plot_weight('P3/cW.csv')

plt.subplot(247)
plt.title('(c) Learning curve')
plot_error('P3/cE.csv')

#-----#

# 4 clusters
plt.subplot(244)
plt.title('(d) 4 learned clusters')
plot_cluster('P3/d.csv', [-1,5], [-3,3])
plot_weight('P3/dW.csv')

plt.subplot(248)
plt.title('(d) Learning curve')
plot_error('P3/dE.csv')

plt.savefig('P3/P3.jpg')

#----------#

# P4

plt.figure(figsize = (16,8))
plt.suptitle('P4: Gaussian clusters, $N_1= N_3 = 100$, $N_2 = 200$')

# Generated data
plt.subplot(241)
plt.title('(a) Generated data')
plot_cluster('P4/P4.csv', [-3,7], [-5,5])
i = 0
for m in Mean4:
	plt.scatter(m[0], m[1], color = palette[i], marker = '^')
	i += 1

#-----#

# 3 clusters
plt.subplot(242)
plt.title('(b) 3 learned clusters')
plot_cluster('P4/b.csv', [-3,7], [-5,5])
plot_weight('P4/bW.csv')

plt.subplot(246)
plt.title('(b) Learning curve')
plot_error('P4/bE.csv')

#-----#

# 2 clusters
plt.subplot(243)
plt.title('(c) 2 learned clusters')
plot_cluster('P4/c.csv', [-3,7], [-5,5])
plot_weight('P4/cW.csv')

plt.subplot(247)
plt.title('(c) Learning curve')
plot_error('P4/cE.csv')

#-----#

# 4 clusters
plt.subplot(244)
plt.title('(d) 4 learned clusters')
plot_cluster('P4/d.csv', [-3,7], [-5,5])
plot_weight('P4/dW.csv')

plt.subplot(248)
plt.title('(d) Learning curve')
plot_error('P4/dE.csv')

plt.savefig('P4/P4.jpg')

'''