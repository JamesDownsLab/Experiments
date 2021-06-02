import time
import numpy as np
from labequipment import shaker

START = 750
END = 550
INT = 25

my_shaker = shaker.Shaker()

last_duty = 775
for duty in np.arange(START, END-INT, -INT):
    my_shaker.ramp(last_duty, duty, 0.5, record=False, stop_at_end=False)
    my_shaker.init_duty(duty)
    time.sleep(5)
    my_shaker.init_duty(duty)
    last_duty = duty

my_shaker.quit()

