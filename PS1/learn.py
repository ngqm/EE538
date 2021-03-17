import pandas as pd

from som import SOM


# LEARN AND SAVE DATA


for problem in range(4):

	n_cluster = [3, 2, 4]
	subproblem = ['b', 'c', 'd']

	p = str(problem + 1)

	for i, j in zip(n_cluster, subproblem):

		model = SOM(i)
		model.learn('P' + p + '/' + 'P' + 
			p + '.csv')
		# learned clusters
		pd.DataFrame(model.clusters).to_csv('P' + 
			p + '/' + j + '.csv',
			header = None, index = None)
		# error log
		pd.DataFrame(model.error).to_csv('P' + 
			p + '/' + j + 'E.csv',
			header = None, index = None)
		# learned centers
		pd.DataFrame(model.weights).to_csv('P' + 
			p + '/' + j + 'W.csv',
			header = None, index = None)

	print('Problem {} done!'.format(p))