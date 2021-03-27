import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from P2d import f 
from P2b import u, T, adjusted_u

np.random.seed(0) # for reproducibility


def sum_f(Tau, Theta, v):
	"""
	total spikes per unit time, defined
	on array of theta and tau
	"""

	F = [f(tau, theta, v) for tau, theta in zip(Tau, Theta)]

	return np.sum(F)


# means
Theta = [15, 20, 25]		# mV
Tau = [8, 10]				# ms

Theta, Tau = np.meshgrid(Theta, Tau)
Theta = Theta.flatten()
Tau = Tau.flatten()

'''
# SUM PLOT

plt.figure(figsize = (7,5))
plt.title('(e) Total firing frequency as a function of $V_i$')
plt.xlabel('Input potential $V_i$ (mV)')
plt.ylabel('Total firing frequency $\Sigma f$ ($ms^{-1}$)')

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


# INDIVIDUAL PLOT

plt.figure(figsize = (18,10))
plt.suptitle('Individual firing frequencies as a function of $V_i$')

for i in range(6):

	tau, theta = Tau[i], Theta[i]

	Tau_array = []

	plt.subplot(2, 3, i+1)
	plt.xlabel('Input potential $V_i$ (mV)')
	plt.ylabel('Firing frequency $f$ ($ms^{-1}$)')
	plt.title(f'$\\bar \\tau_m$ = {tau} ms, $\\bar \\theta$ = {theta} mV')

	while len(Tau_array) < 200:
	
		random_tau = tau/10*np.random.randn() + tau
		if random_tau > 0:
			Tau_array.append(random_tau)

	Theta_array = theta/10*np.random.randn(200) + theta
	
	V = np.arange(0, 40, .2)	# mV
	
	for j in range(200):

		Frequency = [f(Tau_array[j], Theta_array[j], v) for v in V]
		# plot individual neurons
		plt.plot(V, Frequency, alpha = .1, color = 'g')
	
plt.savefig('P2ef.jpg')
'''


# MEMBRANE POTENTIAL INDIVIDUAL PLOT

# V = 30 mV
plt.figure(figsize = (18,10))
plt.suptitle('Individual membrane potential as a function of time')

for i in range(6):

	tau, theta = Tau[i], Theta[i]

	Tau_array = []

	plt.subplot(2, 3, i+1)
	plt.xlabel('Time $t$ (ms)')
	plt.ylabel('Membrane potential $u$ ($mV$)')
	plt.title(f'$\\bar \\tau_m$ = {tau} ms, $\\bar \\theta$ = {theta} mV, $V_i$ = 30 mV')

	while len(Tau_array) < 200:
	
		random_tau = tau/10*np.random.randn() + tau
		if random_tau > 0:
			Tau_array.append(random_tau)

	Theta_array = theta/10*np.random.randn(200) + theta
	
	v = 30	# mV
	
	for j in range(200):

		Time = np.arange(0, 3.5*T(tau, theta, v), .05)
		U = adjusted_u(Tau_array[j], Theta_array[j], v, Time)
		# plot individual neurons
		plt.plot(Time, U, alpha = .05, color = 'g')
	plt.plot(Time, adjusted_u(tau, theta, v, Time),
		label = 'Average neuron', color = 'black')
	plt.legend()
	
plt.savefig('P2eu30.jpg')



# V = 35 mV
plt.figure(figsize = (18,10))
plt.suptitle('Individual membrane potential as a function of time')

for i in range(6):

	tau, theta = Tau[i], Theta[i]

	Tau_array = []

	plt.subplot(2, 3, i+1)
	plt.xlabel('Time $t$ (ms)')
	plt.ylabel('Membrane potential $u$ ($mV$)')
	plt.title(f'$\\bar \\tau_m$ = {tau} ms, $\\bar \\theta$ = {theta} mV, $V_i$ = 35 mV')

	while len(Tau_array) < 200:
	
		random_tau = tau/10*np.random.randn() + tau
		if random_tau > 0:
			Tau_array.append(random_tau)

	Theta_array = theta/10*np.random.randn(200) + theta
	
	v = 35	# mV
	
	for j in range(200):

		Time = np.arange(0, 3.5*T(tau, theta, v), .05)
		U = adjusted_u(Tau_array[j], Theta_array[j], v, Time)
		# plot individual neurons
		plt.plot(Time, U, alpha = .05, color = 'g')
	plt.plot(Time, adjusted_u(tau, theta, v, Time),
		label = 'Average neuron', color = 'black')
	plt.legend()
	
plt.savefig('P2eu35.jpg')