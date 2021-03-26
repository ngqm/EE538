import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from P2d import f 


def sum_f(Tau, Theta, v):
	"""
	total spikes per unit time, defined
	on array of theta and tau
	"""

	F = [f(tau, theta, v) for tau, theta in zip(Tau, Theta)]

	return np.sum(F)


# the means
Theta = [15, 20, 25]		# mV
Tau = [8, 10]				# ms

Theta, Tau = np.meshgrid(Theta, Tau)
Theta = Theta.flatten()
Tau = Tau.flatten()

plt.figure(figsize = (7,5))
plt.title('(e) Total firing frequency as a function of $V_i$')
plt.xlabel('$V_i$ (mV)')
plt.ylabel('Total firing frequency $\Sigma f$ ($ms^{-1}$)')

epsilon = 10e-6

for i in range(6):

	tau, theta = Tau[i], Theta[i]

	Tau_array = []

	while len(Tau_array) < 200:
	
		random_tau = tau/10*np.random.randn() + tau
		if random_tau > 0:
			Tau_array.append(random_tau)

	Theta_array = theta/10*np.random.randn(200) + theta
	
	V = np.arange(0, 40, .2)	# mV
	Frequency = [sum_f(Tau_array, Theta_array, v) for v in V]

	# plot
	plt.plot(V, Frequency, label = f'$\\bar \\tau_m$ = {tau} ms, $\\bar \\theta$ = {theta} mV')
	plt.legend()
	
plt.savefig('P2e.jpg')