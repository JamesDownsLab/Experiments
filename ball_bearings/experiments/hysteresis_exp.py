import time

from labequipment import shaker

rates = [0.4, 0.8]
REPEATS = [5, 5]

TOP = 530
BOTTOM = 400

s = shaker.Shaker()
s.change_duty(TOP)
time.sleep(15)
#
for rate, repeats in zip(rates, REPEATS):
    for repeat in range(repeats):
        s.ramp(TOP, BOTTOM, rate, record=True)
        time.sleep(5)
        s.ramp(BOTTOM, TOP, rate, record=True)
        time.sleep(60)
    s.change_duty(TOP)
    time.sleep(20)

s.change_duty(0)