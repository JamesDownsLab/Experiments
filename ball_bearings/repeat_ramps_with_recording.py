from labequipment import shaker
import numpy as np
import time

s = shaker.Shaker()
s.change_duty(550)

for i in range(50):
    s.ramp(550, 450, 0.2)
    s.init_duty(450)
    time.sleep(5)
    s.init_duty(450)
    time.sleep(3)

