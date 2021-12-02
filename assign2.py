import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.markersize'] = 0.5
np.random.seed(1)

method = 'leapFrog' # can be 'modifiedEuler', 'leapFrog', 'SS3'
numPts = 100 # how many timesteps will be taken
omega = 1. # angular frequency of oscillator
h = 0.6 # size of timestep

def mod2pi(theta): # places values within [0, 2pi)
    pi2 = 2*np.pi
    return np.fmod((pi2 + np.fmod(theta, pi2)), pi2)

# Potential function = -omega**2 * sin(q)
# p - momentum of system
# q - position of system

def modifiedEuler(p, q):
    p = p - omega**2 * h * np.sin(q) # kick one time step
    q = q + h * p # drift one time step
    return p, q

def leapFrog(p, q, const):
    # const is for split 
    qprime = q + 0.5*const * h * p # drift half time step
    p = p - omega**2*const * h * np.sin(qprime) # kick full time step
    q = qprime + 0.5*const* h * p # drift another half time step
    return p, q

def SS3(p, q):
    alpha = 1 / (2 - 2**(1/3.))
    beta = 1 - 2*alpha

    p, q = leapFrog(p, q, alpha) # S_alpha
    p, q = leapFrog(p, q, beta) # S_beta
    p, q = leapFrog(p, q, alpha) # S_alpha
    return p, q

p = np.linspace(-np.pi, np.pi, int(numPts/2.))
q = np.copy(p) # initial conditions
p_arr = np.zeros((len(p), numPts))
q_arr = np.zeros((len(q), numPts)) # arrays to be filled for plotting
p_arr[:,0] = p
q_arr[:,0] = q # save initial conditions to plot array

for i in range(1,numPts): # evolve over numPts timesteps
    if method == 'modifiedEuler':
        p, q = modifiedEuler(p, q)
    if method == 'leapFrog':
        p, q = leapFrog(p, q, 1.)
    if method == 'SS3':
        p, q = SS3(p, q)
    p_arr[:,i] = p
    q_arr[:,i] = q # save updated values to be plotted

plt.plot(mod2pi(q_arr + np.pi), mod2pi(p_arr), '.k')
plt.xlim(0, 2*np.pi)
plt.ylim(0, 2*np.pi)
plt.xticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'2$\pi$'])
plt.yticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'2$\pi$'])
plt.title(method + ', h = ' + str(h))
plt.xlabel('q')
plt.ylabel('p')
plt.savefig(method + '_' + str(h) + '.png')
plt.clf()