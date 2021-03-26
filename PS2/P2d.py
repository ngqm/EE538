import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def f(tau, theta, v):
	"""
	firing frequency as a function of 
	potential parameter
	"""
	if v > theta:
		return (tau*np.log(v/(v-theta)))**(-1)
	else:
		return 0


Theta = [15, 20, 25]		# mV
Tau = [8, 10]				# ms

Theta, Tau = np.meshgrid(Theta, Tau)
Theta = Theta.flatten()
Tau = Tau.flatten()

plt.figure(figsize = (7,5))
plt.title('(d) Firing frequency as a function of $V_i$')
plt.xlabel('$V_i$ (mV)')
plt.ylabel('Firing frequency $f$ ($ms^{-1}$)')

for i in range(6):

	tau, theta = Tau[i], Theta[i]

	V = np.arange(0, 40, .2)	# mV
	Frequency = [f(tau, theta, v) for v in V]

	# plot
	plt.plot(V, Frequency, label = f'$\\tau_m = {tau}$ ms, $\\theta$ = {theta} mV')
	plt.legend()

plt.savefig('P2d.jpg')