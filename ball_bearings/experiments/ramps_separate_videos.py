import time

from labequipment import shaker

s = shaker.Shaker()
s.change_duty(550)
time.sleep(3)

for d in range(550, 450, -1):
    # s.change_duty(d)
    time.sleep(10)
    s.init_duty(d)
    time.sleep(3)
    s.init_duty(d-1)

time.sleep(3)
s.change_duty(0)