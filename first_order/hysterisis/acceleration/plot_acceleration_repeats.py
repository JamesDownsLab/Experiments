import os

import numpy as np
import matplotlib.pyplot as plt



direc = "/media/data/Data/FirstOrder/Hysterisis/AccelerationHysterisis/repeat_data/"

heat = np.loadtxt(direc+'heat.txt')
cool = np.loadtxt(direc+'cool.txt')

cool_V = np.array([np.loadtxt(f'{direc}cool_V_{i}.txt') for i in range(20)])
heat_V = np.array([np.loadtxt(f'{direc}heat_V_{i}.txt') for i in range(20)])

for cV, hV in zip(cool_V, heat_V):
    plt.plot(heat, hV)
    plt.plot(cool, cV)

cool_V_mean = np.mean(cool_V, axis=0)
cool_V_err = np.std(cool_V, axis=0)

heat_V_mean = np.mean(heat_V, axis=0)
heat_V_err = np.std(heat_V, axis=0)

plt.figure()
plt.errorbar(cool, cool_V_mean, cool_V_err, fmt='x', label='cool')
plt.errorbar(heat, heat_V_mean, heat_V_err, fmt='.', label='heat')
plt.legend(frameon=False)
plt.xlabel('Duty Cycle')
plt.ylabel('Accelerometer Reading (V)')
plt.title('Averages of acceleration over 20 ramps to test hysterisis in acceleration')

p, v = np.polyfit(cool, cool_V_mean, 1, w=1/cool_V_err, cov=True)
print(f'V = ({p[0]} +/- {v[0, 0]})D + ({p[1]} +/- {v[1, 1]})')
plt.plot(cool, cool*p[0] + p[1])



plt.show()