from labequipment import shaker
import time

s = shaker.Shaker()
s.change_duty(550)
time.sleep(5)
s.ramp(550, 440, 1)

for d in range(450, 550, 5):
    s.ramp(440, d, 1)
    s.init_duty(d)
    time.sleep(5)
    s.init_duty(d)
    s.ramp(d, 440, 0.1)
    s.init_duty(440)
    time.sleep(5)
    s.init_duty(440)
    time.sleep(5)

s.change_duty(0)