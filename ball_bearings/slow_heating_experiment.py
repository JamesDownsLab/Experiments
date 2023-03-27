from labequipment import shaker
import time

s = shaker.Shaker()
s.change_duty(440)
# s.ramp(550, 490, 1)

for d in range(440, 500, 1):
    s.change_duty(d)
    time.sleep(30)
    s.init_duty(d)
    time.sleep(5)
    s.init_duty(d+1)
s.change_duty(0)