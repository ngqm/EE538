import numpy as np 


def competitive(data, n_cluster):
	"""
	Receive data and output centers
	"""

	eta0 = .2 # learning rate
	w = np.random.rand(n_cluster, 2)

	for epoch in range(5):
		eta = eta0/(epoch + 1)
		epoch_data = np.random.permutation(data)
		for x in epoch_data:
			i = np.argmin(np.linalg.norm(x-w, axis=1))
			w[i] += eta*(x-w[i])

	return w

def color_code(data, w):
	"""
	Predict cluster for data and 
	return n_cluster lists
	"""
	code = []
	for i in range(len(w)):
		new_code = [x for x in data if np.argmin(np.linalg.norm(x-w, axis=1))==i]
		new_code = np.array(new_code)
		code.append(new_code)

	return np.array(code)