from labequipment import shaker
import time

s = shaker.Shaker()
for repeat in range(20):
    s.ramp(540, 460, 0.5)
    s.init_duty(460)
    time.sleep(5)
    s.init_duty(460)
    time.sleep(5)
    