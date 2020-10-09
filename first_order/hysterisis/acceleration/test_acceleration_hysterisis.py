import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from labequipment import shaker, picoscope

direc = "/media/data/Data/FirstOrder/Hysterisis/AccelerationHysterisis/data"

# RATE = 0.1
# START = 700
# END = 600
# INTERVAL = 1 / RATE
# s = shaker.Shaker()
# scope = picoscope.Picoscope()
#
# down = np.arange(START, END-1, -1)
# down_v = []
# down_v_std = []
#
# for i, d in tqdm(enumerate(down)):
#     t = time.time()
#     s.change_duty(d)
#     v_rms = [np.mean(scope.get_V()[1]**2) for r in range(10)]
#     down_v.append(np.mean(v_rms))
#     down_v_std.append(np.std(v_rms))
#     elapsed = time.time() - t
#     time.sleep(INTERVAL - elapsed)
#
# up = np.arange(END, START+1)
# up_v = []
# up_v_std = []
# for j, u in tqdm(enumerate(up)):
#     t = time.time()
#     s.change_duty(u)
#     v_rms = [np.mean(scope.get_V()[1]**2) for r in range(10)]
#     up_v.append(np.mean(v_rms))
#     up_v_std.append(np.std(v_rms))
#     elapsed = time.time() - t
#     time.sleep(INTERVAL - elapsed)
#
# np.savetxt(f'{direc}/up_voltage.txt', up_v)
# np.savetxt(f'{direc}/up_voltage_err.txt', up_v_std)
# np.savetxt(f'{direc}/up_duty.txt', up)
#
# np.savetxt(f'{direc}/down_voltage.txt', down_v)
# np.savetxt(f'{direc}/down_voltage_err.txt', down_v_std)
# np.savetxt(f'{direc}/down_duty.txt', down)

up_v = np.loadtxt(f'{direc}/up_voltage.txt')
up_v_std = np.loadtxt(f'{direc}/up_voltage_err.txt')
up = np.loadtxt(f'{direc}/up_duty.txt')

down_v = np.loadtxt(f'{direc}/down_voltage.txt')
down_v_std = np.loadtxt(f'{direc}/down_voltage_err.txt')
down = np.loadtxt(f'{direc}/down_duty.txt')


print(down_v)
plt.errorbar(down, down_v, down_v_std, label='cooling')
plt.errorbar(up, up_v, up_v_std, label='heating')
plt.xlabel('Duty')
plt.ylabel('Accelerometer Voltage (V)')
plt.legend()
plt.show()