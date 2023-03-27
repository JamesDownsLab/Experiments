from labequipment import shaker
import numpy as np
import time

START = 550
END = 420
RATES = [1]#, 0.5, 0.1]

s = shaker.Shaker()
s.change_duty(START)

for rate in RATES:
    s.ramp(START, END, rate, record=True)
    time.sleep(5)
    # s.change_duty(START)
    # time.sleep(5)
