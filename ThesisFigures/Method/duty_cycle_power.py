import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.05, 500)

V = np.sin(2*np.pi*50*t)
I = np.sin(2*np.pi*50*t+0.4*np.pi)
P = I*V

plt.plot(t, V)
plt.plot(t, I)
plt.plot(t, np.abs(P))
plt.show()

energy_time_inst = []
duties = []
for Pi, ti in zip(P[:250], t[:250]):
    energy_time_inst.append(np.abs(Pi)*(t[1]-t[0]))
    duties.append(ti/0.025*100)

energy_time = np.cumsum(energy_time_inst)

plt.figure()
plt.plot(duties, energy_time)
plt.show()