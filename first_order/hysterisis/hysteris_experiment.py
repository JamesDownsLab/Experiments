import sys
from labequipment import shaker
import time

# start, end, rate, delay = sys.argv[1:5]
# start = int(start)
# end = int(end)
# rate = float(rate)
# delay = int(delay)

start = 700
end = 600
rate = 0.2
delay = 10

s = shaker.Shaker()
for i in range(5):
    s.change_duty(start)
    time.sleep(5)
    s.ramp(start, end, rate, record=True)
    time.sleep(delay)
    s.ramp(end, start, rate, record=True)
    time.sleep(20)

# rate = 0.2
# for i in range(5):
#     s.change_duty(start)
#     time.sleep(5)
#     s.ramp(start, end, rate, record=True)
#     time.sleep(delay)
#     s.ramp(end, start, rate, record=True)
#     time.sleep(20)
#
# s.change_duty(0)