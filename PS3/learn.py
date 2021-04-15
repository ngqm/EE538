import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from generate import generate_gaussian

np.random.seed(0)


# SAMPLING 

# modifiable parameters
N1, N2 = 1000, 1000
sigma1, sigma2 = 1, np.sqrt(2)
mean1, mean2 = [1,2], [0,0]

cov1 = sigma1**2*np.identity(2)
cov2 = sigma2**2*np.identity(2)


# gaussian
A_train = generate_gaussian(mean1, cov1, N1)
B_train = generate_gaussian(mean2, cov2, N2)


'''
# uniform
A_train = []
while len(A_train) < N1:
	x, y = 2*np.random.rand(2) - 1
	if x**2 + y**2 < 1:
		point = 2*np.sqrt(2)*sigma1*np.array([x,y]) + mean1
		A_train.append(point)
B_train = []
while len(B_train) < N2:
	x, y = 2*np.random.rand(2) - 1
	if x**2 + y**2 < 1:
		point = 2*np.sqrt(2)*sigma2*np.array([x,y]) + mean2
		B_train.append(point)

A_train, B_train = np.array(A_train), np.array(B_train)
'''

train = np.concatenate((A_train, B_train))
train_label = np.append([1]*N1, [-1]*N2)


# gaussian
A_test = generate_gaussian(mean1, cov1, 2*N1)
B_test = generate_gaussian(mean2, cov2, 2*N2)

'''
# uniform
A_test = []
while len(A_test) < 3*N1:
	x, y = 2*np.random.rand(2) - 1
	if x**2 + y**2 < 1:
		point = np.sqrt(2)*sigma1*np.array([x,y]) + mean1
		A_test.append(point)
B_test = []
while len(B_test) < 3*N2:
	x, y = 2*np.random.rand(2) - 1
	if x**2 + y**2 < 1:
		point = np.sqrt(2)*sigma2*np.array([x,y]) + mean2
		B_test.append(point)

A_test, B_test = np.array(A_test), np.array(B_test)
'''

test = np.concatenate((A_test, B_test))
test_label = np.append([1]*3*N1, [-1]*3*N2)


##########


# LEARNING

# modifiable parameters
lr0 = .2 # initial learning rate
max_epoch = 15
w = 0.2*np.random.rand(3)

def accuracy(w, data, label):
	correct = sum([1 for x, y in zip(data, label) if y*w.dot(x) >= 0])
	return correct/len(data)

def unison_shuffles(a, b):
	p = np.random.permutation(len(a))
	return a[p], b[p]

data = np.array([np.append(x,1) for x in train])

accuracy_log = [[accuracy(w, data, train_label)]*2]
a = np.array(w)

print('learning, please wait...')

for epoch in range(max_epoch):
	
	lr = lr0/(epoch + 1)
	epoch_data, label = unison_shuffles(data, train_label)
	iteration = 0
	for x, y in zip(epoch_data, label):
		iteration += 1
		print(f'Epoch {epoch+1}, iteration {iteration}', end='\r')
		if y*w.dot(x) < 0:
			w += lr*y*x # learning rule
			a += w

	accuracy_log.append([accuracy(w, epoch_data, label), 
		accuracy(a, epoch_data, label)])

accuracy_log = np.array(accuracy_log)


# PLOT

delta = 0.025
rangex = np.arange(-6, 7.5, delta)
rangey = np.arange(-4, 8, delta)
X, Y = np.meshgrid(rangex,rangey)

test_data = np.array([np.append(x,1) for x in test])

plt.figure(figsize = (14, 14))

plt.suptitle(f'(cd) Gaussian distributions with $\sigma_1$ = {sigma1}, $\sigma_2$ = sqrt(2), $N_1$ = {N1}, $N_2$ = {N2}',
	size = 'xx-large')

plt.subplot(221) # decision boundary of Bayes classifier

# F = 0 is implicit function
F = (X - 2)**2 + (Y - 4)**2 - 4*np.log(2) - 10

plt.title('Bayes classifier decision boundary')
plt.contour(X, Y, F, [0], colors = 'black')
plt.scatter(A_test[:,0], A_test[:,1], alpha=.5, 
	s=15, label = 'Class A')
plt.scatter(B_test[:,0], B_test[:,1], alpha=.5, 
	s=15, label = 'Class B')
plt.legend(loc = 'lower left')

plt.subplot(222) # decision boundary of perceptron

# F = 0 is implicit function
F = a[0]*X + a[1]*Y + a[2]

accu = accuracy(a, test_data, test_label)
plt.title('Perceptron decision boundary ({:.3f})'.format(accu))
plt.contour(X, Y, F, [0.], colors = 'black')
plt.scatter(A_test[:,0], A_test[:,1], alpha=.5, 
	s=15, label = 'Class A')
plt.scatter(B_test[:,0], B_test[:,1], alpha=.5, 
	s=15, label = 'Class B')
plt.legend(loc = 'lower left')

plt.subplot(224)

plt.title('Learning curve')
plt.xlabel('Epoch')
plt.ylabel('Train set accuracy')
plt.plot(range(max_epoch+1), accuracy_log[:,0], 
	label = 'Learning perceptron')
plt.plot(range(max_epoch+1), accuracy_log[:,1], 
	label = 'Average perceptron')
plt.legend(loc = 'lower left')

plt.savefig('gaussian.jpg')