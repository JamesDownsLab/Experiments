from labequipment import shaker
import time
import numpy as np

s = shaker.Shaker()

s.change_duty(700)

for r in range(100):
    s.ramp(700, 630, 0.2)
    s.init_duty(630)
    time.sleep(30)
    s.init_duty(630)
    time.sleep(15)

s.quit()