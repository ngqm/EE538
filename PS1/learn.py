import pandas as pd

from som import SOM


# P1

# 3 clusters
P1b = SOM(n_cluster = 3)
P1b.learn('P1/P1.csv')
pd.DataFrame(P1b.clusters).to_csv('P1/b.csv', 
	header = None, index = None)
pd.DataFrame(P1b.error).to_csv('P1/bE.csv',
	header = None, index = None)
pd.DataFrame(P1b.weights).to_csv('P1/bW.csv',
	header = None, index = None)

# 2 clusters
P1c = SOM(n_cluster = 2)
P1c.learn('P1/P1.csv')
pd.DataFrame(P1c.clusters).to_csv('P1/c.csv', 
	header = None, index = None)
pd.DataFrame(P1c.error).to_csv('P1/cE.csv',
	header = None, index = None)
pd.DataFrame(P1c.weights).to_csv('P1/cW.csv',
	header = None, index = None)

# 4 clusters
P1d = SOM(n_cluster = 4)
P1d.learn('P1/P1.csv')
pd.DataFrame(P1d.clusters).to_csv('P1/d.csv', 
	header = None, index = None)
pd.DataFrame(P1d.error).to_csv('P1/dE.csv',
	header = None, index = None)
pd.DataFrame(P1d.weights).to_csv('P1/dW.csv',
	header = None, index = None)


#-----#


# P3

# 3 clusters
P3b = SOM(n_cluster = 3)
P3b.learn('P3/P3.csv')
pd.DataFrame(P3b.clusters).to_csv('P3/b.csv', 
	header = None, index = None)
pd.DataFrame(P3b.error).to_csv('P3/bE.csv',
	header = None, index = None)
pd.DataFrame(P3b.weights).to_csv('P3/bW.csv',
	header = None, index = None)

# 2 clusters
P3c = SOM(n_cluster = 2)
P3c.learn('P3/P3.csv')
pd.DataFrame(P3c.clusters).to_csv('P3/c.csv', 
	header = None, index = None)
pd.DataFrame(P3c.error).to_csv('P3/cE.csv',
	header = None, index = None)
pd.DataFrame(P3c.weights).to_csv('P3/cW.csv',
	header = None, index = None)

# 4 clusters
P3d = SOM(n_cluster = 4)
P3d.learn('P3/P3.csv')
pd.DataFrame(P3d.clusters).to_csv('P3/d.csv', 
	header = None, index = None)
pd.DataFrame(P3d.error).to_csv('P3/dE.csv',
	header = None, index = None)
pd.DataFrame(P3d.weights).to_csv('P3/dW.csv',
	header = None, index = None)