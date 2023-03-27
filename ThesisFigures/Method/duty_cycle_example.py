import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.02, 50)
V = 230*np.sin(2*np.pi*50*t)

t1 = 0.0025
t2 = 0.005
t3 = 0.0075

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=True)

ax1.plot(t, V, 'k')
ax1.axhline(0, ls='--', c='k')
ax1.set_ylabel('Voltage (V)')


ax2.plot(t[t < t1], V[t < t1], 'k')
ax2.plot(t[t >= t1], V[t >= t1], 'k--', alpha=0.5)
ax2.fill_between(t[t < t1], 0, V[t < t1], color='g')
ax2.axhline(0, ls='--', c='k')
ax2.axvline(0.0025, label='25%', c='g')
ax2.legend()


ax3.plot(t[t < t2], V[t < t2], 'k')
ax3.plot(t[t >= t2], V[t >= t2], 'k--', alpha=0.5)
ax3.fill_between(t[t < t2], 0, V[t < t2], color='m')
ax3.axhline(0, ls='--', c='k')
ax3.axvline(0.005, label='50%', c='m')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Voltage (V)')
ax3.legend()


ax4.plot(t[t < t3], V[t < t3], 'k')
ax4.plot(t[t >= t3], V[t >= t3], 'k--', alpha=0.5)
ax4.fill_between(t[t < t3], 0, V[t < t3], color='r')
ax4.axhline(0, ls='--', c='k')
ax4.axvline(0.0075, label='75%', c='r')
ax4.legend()
ax4.set_xlabel('Time (s)')
plt.savefig('Duty Cycle.jpg')