from labequipment import shaker
import numpy as np
import time

START = 520
END = 420
REPEATS = 10
RATE = 0.2
RECORD = 5
WAIT = 3

s = shaker.Shaker()
s.change_duty(START)

for i in range(REPEATS):
    s.ramp(START, END, RATE)
    s.init_duty(END)
    time.sleep(RECORD)
    s.init_duty(END)
    time.sleep(WAIT)

