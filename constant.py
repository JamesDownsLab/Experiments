import sys
from labequipment import shaker
import time

duty, duration = sys.argv[1:3]
duty = int(duty)
duration = int(duration)

s = shaker.Shaker()
s.init_duty(duty)
time.sleep(duration)
s.init_duty(duty)
s.quit()

