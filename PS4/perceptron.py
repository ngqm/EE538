import numpy as np 

np.random.seed(0)

ranval = 0.2*np.random.rand(3) - 0.1

def accuracy(w, data, label):
	correct = sum([1 for x, y in zip(data, label) if y*w.dot(x) >= 0])
	return correct/len(data)

def unison_shuffles(a, b):
	p = np.random.permutation(len(a))
	return a[p], b[p]

def perceptron(data, label, eta0=.3, init_mean=0):

	eta0 = eta0
	w = 0.4*np.random.rand(3) - 0.2
	#w = np.array(ranval)
	
	train_data = np.array([np.append(x,1) for x in data])
	accuracy_log = [[accuracy(w, train_data, label)]*2]
	a = np.array(w)

	for epoch in range(10):

		eta = eta0/(epoch+1)
		epoch_data, epoch_label = unison_shuffles(train_data, label)
		for x, y in zip(epoch_data, epoch_label):
			if y*w.dot(x) < 0:
				w += eta*y*x # learning rule
				a += w

		accuracy_log.append([accuracy(w, epoch_data, epoch_label), 
			accuracy(a, epoch_data, epoch_label)])

	accuracy_log = np.array(accuracy_log)

	return w, accuracy_log

def pcolor_code(data, a):
	"""
	Predict class for data and 
	return 2 lists
	"""
	train_data = np.array([np.append(x,1) for x in data])
	code1 = []
	code0 = []
	for x in train_data:
		if a.dot(x) < 0:
			code0.append(x)
		else:
			code1.append(x)
	code0, code1 = np.array(code0), np.array(code1)

	return np.array([code0, code1])