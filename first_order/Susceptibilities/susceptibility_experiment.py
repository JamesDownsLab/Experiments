from labequipment import shaker
import time
import numpy as np
from tqdm import tqdm

s = shaker.Shaker()

duties = np.arange(680, 580, -1)
for d in tqdm(duties):
    s.init_duty(d)
    time.sleep(30)
    s.init_duty(d)
    time.sleep(10)

s.quit()