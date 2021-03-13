import pandas as pd

from som import SOM


# P0
# using same data as P1, but no cooperation

# 3 clusters
P0b = SOM(n_cluster = 3, cooperative = False)
P0b.learn('P1/P1.csv')
pd.DataFrame(P0b.clusters).to_csv('P0/b.csv', 
	header = None, index = None)
pd.DataFrame(P0b.error).to_csv('P0/bE.csv',
	header = None, index = None)
pd.DataFrame(P0b.weights).to_csv('P0/bW.csv',
	header = None, index = None)

# 2 clusters
P0c = SOM(n_cluster = 2, cooperative = False)
P0c.learn('P1/P1.csv')
pd.DataFrame(P0c.clusters).to_csv('P0/c.csv', 
	header = None, index = None)
pd.DataFrame(P0c.error).to_csv('P0/cE.csv',
	header = None, index = None)
pd.DataFrame(P0c.weights).to_csv('P0/cW.csv',
	header = None, index = None)

# 4 clusters
P0d = SOM(n_cluster = 4, cooperative = False)
P0d.learn('P1/P1.csv')
pd.DataFrame(P0d.clusters).to_csv('P0/d.csv', 
	header = None, index = None)
pd.DataFrame(P0d.error).to_csv('P0/dE.csv',
	header = None, index = None)
pd.DataFrame(P0d.weights).to_csv('P0/dW.csv',
	header = None, index = None)


'''
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

# P2

# 3 clusters
P2b = SOM(n_cluster = 3)
P2b.learn('P2/P2.csv')
pd.DataFrame(P2b.clusters).to_csv('P2/b.csv', 
	header = None, index = None)
pd.DataFrame(P2b.error).to_csv('P2/bE.csv',
	header = None, index = None)
pd.DataFrame(P2b.weights).to_csv('P2/bW.csv',
	header = None, index = None)

# 2 clusters
P2c = SOM(n_cluster = 2)
P2c.learn('P2/P2.csv')
pd.DataFrame(P2c.clusters).to_csv('P2/c.csv', 
	header = None, index = None)
pd.DataFrame(P2c.error).to_csv('P2/cE.csv',
	header = None, index = None)
pd.DataFrame(P2c.weights).to_csv('P2/cW.csv',
	header = None, index = None)

# 4 clusters
P2d = SOM(n_cluster = 4)
P2d.learn('P2/P2.csv')
pd.DataFrame(P2d.clusters).to_csv('P2/d.csv', 
	header = None, index = None)
pd.DataFrame(P2d.error).to_csv('P2/dE.csv',
	header = None, index = None)
pd.DataFrame(P2d.weights).to_csv('P2/dW.csv',
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

#-----#


# P4

# 3 clusters
P4b = SOM(n_cluster = 3)
P4b.learn('P4/P4.csv')
pd.DataFrame(P4b.clusters).to_csv('P4/b.csv', 
	header = None, index = None)
pd.DataFrame(P4b.error).to_csv('P4/bE.csv',
	header = None, index = None)
pd.DataFrame(P4b.weights).to_csv('P4/bW.csv',
	header = None, index = None)

# 2 clusters
P4c = SOM(n_cluster = 2)
P4c.learn('P4/P4.csv')
pd.DataFrame(P4c.clusters).to_csv('P4/c.csv', 
	header = None, index = None)
pd.DataFrame(P4c.error).to_csv('P4/cE.csv',
	header = None, index = None)
pd.DataFrame(P4c.weights).to_csv('P4/cW.csv',
	header = None, index = None)

# 4 clusters
P4d = SOM(n_cluster = 4)
P4d.learn('P4/P4.csv')
pd.DataFrame(P4d.clusters).to_csv('P4/d.csv', 
	header = None, index = None)
pd.DataFrame(P4d.error).to_csv('P4/dE.csv',
	header = None, index = None)
pd.DataFrame(P4d.weights).to_csv('P4/dW.csv',
	header = None, index = None)

'''