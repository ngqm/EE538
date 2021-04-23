import numpy as np 
import matplotlib.pyplot as plt
import itertools

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

np.random.seed(0) # for reproducibility


# GENERATE DATA

N = 500 # number of points per class 

A, B = [], [] # class 1 and class 0

while len(A) < N:
	x, y = [26,13]*np.random.rand(2) - [13,0]
	r2 = x**2 + y**2
	if r2 > 49 and r2 < 169:
		A.append([x,y])
while len(B) < N:
	x, y = [26,13]*np.random.rand(2) - [3,9]
	r2 = (x-10)**2 + (y-4)**2
	if r2 > 49 and r2 < 169:
		B.append([x,y])

A, B = np.array(A), np.array(B)
data = np.concatenate((A,B))
label = np.append([1]*N,[-1]*N)


# VISUALIZE

'''
# Generated data

plt.figure(figsize=(7,7))
plt.title('Generated data')
plt.xlim(-14,24)
plt.ylim(-12,16)

plt.scatter(A[:,0],A[:,1], label='Class 1',
	alpha=.5)
plt.scatter(B[:,0],B[:,1], label='Class 0',
	alpha=.5)

plt.legend()
plt.savefig('vanilla.jpg')
'''


'''
# Competitive learning
from competitive import competitive, color_code

plt.figure(figsize=(14,14))
plt.suptitle('Competitive learning', size = 'xx-large')

n_cluster = [2,3,4,8]

for i, n in enumerate(n_cluster):

	plt.subplot(221+i)
	plt.title(f'{n} clusters')
	plt.xlim(-14,24)
	plt.ylim(-12,16)

	j=0
	w = competitive(data, n)
	for cluster in color_code(data,w):
		plt.scatter(cluster[:,0], cluster[:,1], 
			color = colors[j], alpha=.2)
		plt.scatter(w[j,0], w[j,1], s=250, marker='^',
			color=colors[j])
		j += 1

	# decision regions
	x = np.arange(-14,24,.3)
	y = np.arange(-12,16,.3)
	xx, yy = np.meshgrid(x,y)
	xx, yy = xx.flatten(), yy.flatten()
	mesh_data = [[x,y] for x, y in zip(xx,yy)]
	j=0
	for cluster in color_code(mesh_data, w):
		plt.scatter(cluster[:,0], cluster[:,1], 
			color = colors[j], alpha=.1, marker='+')
		j += 1

plt.savefig('competitive.jpg')
'''

'''
# perceptron


from perceptron import *


# learning curve and decision boundary for 
# different learning rates

plt.figure(figsize = (21,14))
plt.suptitle('Perceptron with different learning rates',
	size = 'xx-large')

learning_rates = [.05, .3, .7]

for i, eta0 in enumerate(learning_rates):

	# decision boundary

	w, log = perceptron(data, label, eta0=eta0)
	plt.subplot(231 + i)
	plt.title(f'Learning rate {eta0}, accuracy {log[-1,0]}')
	plt.xlim(-14,24)
	plt.ylim(-12,16)

	class0, class1 = pcolor_code(data, w)
	plt.scatter(class0[:,0], class0[:,1], 
		color = colors[0], alpha=.3, label='Class 0')
	plt.scatter(class1[:,0], class1[:,1], 
		color = colors[1], alpha=.3, label='Class 1')
	plt.legend()

	# decision regions
	x = np.arange(-14,24,.3)
	y = np.arange(-12,16,.3)
	xx, yy = np.meshgrid(x,y)
	xx, yy = xx.flatten(), yy.flatten()
	mesh_data = [[x,y] for x, y in zip(xx,yy)]
	class0, class1 = pcolor_code(mesh_data, w)
	plt.scatter(class0[:,0], class0[:,1], 
		color = colors[0], alpha=.1, marker='+')
	plt.scatter(class1[:,0], class1[:,1], 
		color = colors[1], alpha=.1, marker='+')

	# learning curve

	plt.subplot(234+i)
	plt.title('Learning curve')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.plot(range(11), log[:,0])

plt.savefig('perceptron.jpg')



# learning curve and decision boundary for 
# different weight init


plt.figure(figsize = (35,14))
plt.suptitle('Perceptron with different weight initializations',
	size = 'xx-large')

for i in range(5):

	# decision boundary

	w, log = perceptron(data, label)
	plt.subplot(2,5,1 + i)
	plt.title(f'Initialization {i+1}, accuracy {log[-1,0]}')
	plt.xlim(-14,24)
	plt.ylim(-12,16)

	class0, class1 = pcolor_code(data, w)
	plt.scatter(class0[:,0], class0[:,1], 
		color = colors[0], alpha=.3, label='Class 0')
	plt.scatter(class1[:,0], class1[:,1], 
		color = colors[1], alpha=.3, label='Class 1')	

	# decision regions
	x = np.arange(-14,24,.3)
	y = np.arange(-12,16,.3)
	xx, yy = np.meshgrid(x,y)
	xx, yy = xx.flatten(), yy.flatten()
	mesh_data = [[x,y] for x, y in zip(xx,yy)]
	class0, class1 = pcolor_code(mesh_data, w)
	plt.scatter(class0[:,0], class0[:,1], 
		color = colors[0], alpha=.1, marker='+')
	plt.scatter(class1[:,0], class1[:,1], 
		color = colors[1], alpha=.1, marker='+')
	plt.legend()

	# learning curve

	plt.subplot(2,5,6+i)
	plt.title('Learning curve')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.plot(range(11), log[:,0])

plt.savefig('perceptron2.jpg')
'''

# Multilayer perceptron

from network import Network
from fclayer import FCLayer
from activation_layer import ActivationLayer
from activations import *
from losses import mse, mse_prime

# training data
x_train = np.array([[x] for x in data])
y_train = np.array([[(y+1)/2] for y in label])

# network
net = Network()
net.add(FCLayer(2, 2))
net.add(ActivationLayer(ReLU, ReLU_prime))
net.add(FCLayer(2, 1))
net.add(ActivationLayer(sigmoid, sigmoid_prime))


plt.figure(figsize = (35,14))
plt.suptitle('(8) 1 hidden layer, 2 neurons each, ReLU nonlinearity with different initializations', 
	size = 'xx-large')

for i in range(5): 

	# train
	net.use(mse, mse_prime)
	net.fit(x_train, y_train, epochs=100, learning_rate=.002)

	# decision boundary

	plt.subplot(2,5,1+i)
	plt.title(f'Initialization {i+1}')
	plt.xlim(-14,24)
	plt.ylim(-12,16)

	class0, class1 = net.mlpcolor_code(x_train)
	plt.scatter(class0[:,0,0], class0[:,0,1], 
		color = colors[0], alpha=.3, label='Class 0')
	plt.scatter(class1[:,0,0], class1[:,0,1], 
		color = colors[1], alpha=.3, label='Class 1')	

	# decision regions
	x = np.arange(-14,24,.3)
	y = np.arange(-12,16,.3)
	xx, yy = np.meshgrid(x,y)
	xx, yy = xx.flatten(), yy.flatten()
	mesh_data = [[x,y] for x, y in zip(xx,yy)]
	mesh_train = np.array([[x] for x in mesh_data])
	class0, class1 = net.mlpcolor_code(mesh_train)
	plt.scatter(class0[:,0,0], class0[:,0,1], 
		color = colors[0], alpha=.1, marker='+')
	plt.scatter(class1[:,0,0], class1[:,0,1],
		color = colors[1], alpha=.1, marker='+')
	plt.legend()


	# learning curve
	plt.subplot(2,5,6+i)
	plt.title('Learning curve')
	plt.xlabel('Epoch')
	plt.ylabel('Cost')
	plt.plot(range(100), net.log)


plt.savefig('mlp82.jpg')