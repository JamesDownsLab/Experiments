from labequipment import shaker
import numpy as np
import time
from tqdm import tqdm

s = shaker.Shaker()

duties = np.arange(690, 590, -5)

d0 = 700
for d in tqdm(duties):
    s.ramp(d0, d, 0.5)
    time.sleep(3)
    s.init_duty(d)
    time.sleep(15)
    s.init_duty(d)
    time.sleep(3)
    d0 = d
