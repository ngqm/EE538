import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from P2d import f 


def sum_f(Tau, Theta, V):
	"""
	total spikes per unit time, defined
	on array of theta and tau
	"""

	# eliminate unphysical time constants
	new_Tau = [tau for tau in Tau if tau > 0]

	# eliminate neurons which never fire
	new_Theta = [theta for theta in Theta if theta < V and theta > 0]

	old_len = len(new_Theta)
	new_len = min(len(new_Tau), len(new_Theta))

	new_Tau = new_Tau[:new_len]
	new_Theta = new_Theta[:new_len]

	return new_len/old_len*np.sum(f(new_Tau, new_Theta, V))


# the means
Theta = [15, 20, 25]		# mV
Tau = [8, 10]				# ms

Theta, Tau = np.meshgrid(Theta, Tau)
Theta = Theta.flatten()
Tau = Tau.flatten()

plt.figure(figsize = (18,10))
plt.suptitle('(e) Total firing frequency as a function of $V_i$')

epsilon = 10e-6

for i in range(6):

	tau, theta = Tau[i], Theta[i]

	Tau_array = tau/10*np.random.randn(200) + tau
	Theta_array = theta/10*np.random.randn(200) + theta
	
	V = np.arange(theta+2,theta+10,.5)	# mV
	
	# plot setup
	plt.subplot(231+i)
	plt.title(f'$\\tau_m$ = {tau} ms, $\\theta$ = {theta} mV')
	plt.xlabel('$V_i$ (mV)')
	plt.ylabel('Firing frequency ($ms^{-1}$)')

	Sum_f = [sum_f(Tau_array, Theta_array, v) for v in V]

	# plot
	plt.plot(V, Sum_f)
	
plt.savefig('P2e.jpg')