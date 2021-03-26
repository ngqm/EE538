import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def u(tau, V, t):
	"""
	membrane potential as a function of time
	"""

	return V*(1-np.exp(-t/tau))


def T(tau, theta, V):
	"""
	the time between consecutive spikes
	"""

	return tau*np.log(V/(V-theta))


def adjusted_u(tau, theta, V, t):
	"""
	membrane potential adjusted for threshold
	"""

	period = T(tau, theta, V)
	adjusted_t = t - (t/period).astype(int)*period

	return u(tau, V, adjusted_t)


Theta = [15, 20, 25]		# mV
Tau = [8, 10]				# ms
V = [30, 35]				# mV

Theta, Tau, V = np.meshgrid(Theta, Tau, V)
Theta = Theta.flatten()
Tau = Tau.flatten()
V = V.flatten()

plt.figure(figsize = (24,15))
plt.suptitle('(b) Membrane potential as a function of time')

for i in range(12):

	theta, tau, v = Theta[i], Tau[i], V[i]

	Time = np.arange(0, 3.5*T(tau, theta, v), .05)

	# plot setup
	plt.subplot(3, 4, i+1)
	plt.title(f'$\\theta$ = {theta} mV, $\\tau_m$ = {tau} ms, $V_i$ = {v} mV')
	plt.xlabel('Time $t$ (ms)')
	plt.ylabel('Membrane potential $u$ (mV)')

	# plot
	plt.plot(Time, adjusted_u(tau, theta, v, Time))

plt.savefig('P2b.jpg')