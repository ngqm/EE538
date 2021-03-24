import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def f(tau, theta, V):
	"""
	firing frequency as a function of 
	potential parameter
	"""

	return (tau*np.log(V/(V-theta)))**(-1)


Theta = [15, 20, 25]		# mV
Tau = [8, 10]				# ms

Theta, Tau = np.meshgrid(Theta, Tau)
Theta = Theta.flatten()
Tau = Tau.flatten()

plt.figure(figsize = (18,10))
plt.suptitle('(d) Firing frequency as a function of $V_i$')

for i in range(6):

	tau, theta = Tau[i], Theta[i]

	V = np.arange(theta+2,theta+10,.5)	# mV
	
	# plot setup
	plt.subplot(231+i)
	plt.title(f'$\\tau_m$ = {tau} ms, $\\theta$ = {theta} mV')
	plt.xlabel('$V_i$ (mV)')
	plt.ylabel('Firing frequency ($ms^{-1}$)')

	# plot
	plt.plot(V, f(tau, theta, V))

plt.savefig('P2d.jpg')