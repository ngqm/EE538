import numpy as np 

def gaussian(x, y, mean, cov):

	sigmax = np.sqrt(cov[0,0])
	sigmay = np.sqrt(cov[1,1])
	mux = mean[0]
	muy = mean[1]

	max_f = (2*np.pi*sigmax*sigmay)**(-1)

	ex = -(x-mux)**2/(2*sigmax**2)
	ey = -(y-muy)**2/(2*sigmay**2)

	return max_f*np.exp(ex + ey)



def generate_gaussian(mean, cov, N):
	'''
	Sample the 2d Gaussian distribution 
	with independent variables.
	'''
	sigmax = np.sqrt(cov[0,0])
	sigmay = np.sqrt(cov[1,1])
	mux = mean[0]
	muy = mean[1]

	max_f = (2*np.pi*sigmax*sigmay)**(-1)

	result = []

	while len(result) < N:

		x = 8*sigmax*np.random.rand() - 4*sigmax + mux
		y = 8*sigmay*np.random.rand() - 4*sigmay + muy
		z = max_f*np.random.rand()

		if z < gaussian(x,y,mean,cov):
			result.append([x,y])

	return np.array(result)