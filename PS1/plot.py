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