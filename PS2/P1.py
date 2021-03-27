import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')
from sklearn.decomposition import PCA

np.random.seed(0) # for reproducibility

sign = lambda x: 1 if x>=0 else -1
def make_first_coordinates_positive(matrix):
	new_rows = []
	for row in matrix:
		new_rows.append(row*sign(row[0]))
	return np.array(new_rows)

# GENERATE DATA

X3 = np.random.rand(600)
X1 = []
X2 = []
while len(X1) < 600:
	x1, x2 = np.random.rand(2)
	if x1**2 + x2**2 < 1:
		X1.append(x1)
		X2.append(x2)

data = np.array([list(point) for point in zip(X1, X2, X3)])

p1, p2, p3 = [[0,0,0]]*200, [[2,0,0]]*200, [[0,2,0]]*200
translation = list(itertools.chain(p1, p2, p3))

data += translation


########## (a)


# FIND EIGENVECTORS WITH sklearn

pca = PCA()
pca.fit(data)
eigen = make_first_coordinates_positive(pca.components_)
mean = pca.mean_

print(f'The eigenvectors are {eigen}')

# VISUALIZE (a)

fig = plt.figure(figsize = (7,5))
ax = Axes3D(fig)
ax.set_title('PCA using sklearn')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')

X, Y, Z = data[:,0], data[:,1], data[:,2]

ax.scatter(X, Y, Z, alpha = .5, color = 'g')

colors = ['r', 'b', 'y']
for i in range(3):
	ax.quiver(mean[0], mean[1], mean[2], 
		eigen[i][0], eigen[i][1], eigen[i][2],
		length = .5, color = colors[i],
		label = f'Eigenvector {i+1}')
plt.legend()

plt.savefig('P1a.jpg')


########## (c)+(e)


# normalize data
normalized_data = data - np.mean(data, axis = 0)

# initialize weights to small values
w1, w2 = .1*np.random.randn(2, 3)
error1 = [np.linalg.norm(w1 - eigen[0])]
error2 = [np.linalg.norm(w1 - eigen[1])]

# initialize learning rate
mu0 = .05

# learn
max_epoch = 5
for epoch in range(max_epoch):

	print(f'Epoch {epoch+1}')

	mu = mu0*np.exp(-epoch) # adaptive learning rate
	
	shuffled_data = np.random.permutation(normalized_data)
	for i, x in enumerate(shuffled_data):	

		y1 = np.dot(w1, x)
		y2 = np.dot(w2, x)
		x_dash = x - y1*w1
		
		# update weights
		w1 += mu*y1*x_dash
		w2 += mu*y2*(x_dash - y2*w2)

		# error log

		temp_w1, temp_w2 = make_first_coordinates_positive([w1, w2])

		error1.append(np.linalg.norm(temp_w1 - eigen[0]))
		error2.append(np.linalg.norm(temp_w2 - eigen[1]))

learned_eigen = make_first_coordinates_positive([w1, w2])

print(f'Learned eigenvectors: {learned_eigen}')



# Learned eigenvectors
fig = plt.figure(figsize = (7,5))
ax = Axes3D(fig)
plt.title('Eigenvectors learned with GHA')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')

ax.scatter(X, Y, Z, alpha = .5, color = 'g')

colors = ['r', 'b']
for i in range(2):
	ax.quiver(mean[0], mean[1], mean[2], 
		learned_eigen[i][0], learned_eigen[i][1], 
		learned_eigen[i][2],
		length = .5, color = colors[i],
		label = f'Eigenvector {i+1}')
plt.legend()

plt.savefig('P1ce.jpg')

# Error curve
plt.figure(figsize = (7,5))

plt.plot(range(max_epoch*600+1), error1, label = 'Error of first component')
plt.plot(range(max_epoch*600+1), error2, label = 'Error of second component')

plt.title('Learning curve of GHA')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()

plt.savefig('P1ceE.jpg')