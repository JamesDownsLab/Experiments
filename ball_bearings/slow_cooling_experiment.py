from labequipment import shaker
import time
from tqdm import tqdm

s = shaker.Shaker()
s.change_duty(540)
# s.ramp(520, 440, 1, record=True)
# # time.sleep(10)

START = 500
START_RECORDING = 480
END = 460
DELAY = 15
RECORD_TIME = 5

#
for d in tqdm(range(START, START_RECORDING, -1)):
    s.change_duty(d)
    time.sleep(DELAY)

for d in tqdm(range(START_RECORDING, END, -1)):
    s.change_duty(d)
    time.sleep(DELAY-RECORD_TIME)
    s.init_duty(d)
    time.sleep(RECORD_TIME)
    s.init_duty(d-1)

for d in tqdm(range(END, START_RECORDING)):
    s.change_duty(d)
    time.sleep(DELAY-RECORD_TIME)
    s.init_duty(d)
    time.sleep(RECORD_TIME)
    s.init_duty(d+1)

