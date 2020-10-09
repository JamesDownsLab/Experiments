from labequipment import shaker
import time
from tqdm import tqdm

s = shaker.Shaker()

for r in tqdm(range(500)):
    s.ramp(700, 640, 1)
    s.init_duty(640)
    time.sleep(10)
    s.init_duty(640)
    time.sleep(10)