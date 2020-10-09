from labequipment import picoscope, shaker
import time
import numpy as np

direc = "/media/data/Data/FirstOrder/Hysterisis/AccelerationHysterisis/repeat_data/"

RATE = 0.1
START = 700
END = 600
INTERVAL = 1 / RATE
REPEATS = 20

my_shaker = shaker.Shaker()
my_scope = picoscope.Picoscope()

for repeat in range(REPEATS):
    cool = np.arange(START, END-1, -1)
    heat = np.arange(END, START+1, 1)
    cool_V = []
    for c in cool:
        t = time.time()
        my_shaker.change_duty(c)
        v_rms = [np.mean(my_scope.get_V()[1]**2) for r in range(10)]
        cool_V.append(np.mean(v_rms))
        elapsed = time.time() - t
        time.sleep(INTERVAL - elapsed)
    np.savetxt(f'{direc}cool_V_{repeat}.txt', cool_V)
    heat_V = []
    for h in heat:
        t = time.time()
        my_shaker.change_duty(h)
        v_rms = [np.mean(my_scope.get_V()[1]**2) for r in range(10)]
        heat_V.append(np.mean(v_rms))
        elapsed = time.time() - t
        time.sleep(INTERVAL - elapsed)
    np.savetxt(f'{direc}heat_V_{repeat}.txt', heat_V)

np.savetxt(f'{direc}heat.txt', heat)
np.savetxt(f'{direc}cool.txt', cool)
