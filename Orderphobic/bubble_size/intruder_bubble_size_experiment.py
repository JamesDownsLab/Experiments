import time

from labequipment import shaker
import numpy as np


s = shaker.Shaker()

s.change_duty(700)

time.sleep(10)
for d in np.arange(700, 600, -1):
    s.init_duty(d)
    time.sleep(10)
    s.init_duty(d)
    time.sleep(10)

s.quit()