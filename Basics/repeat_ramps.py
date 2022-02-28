from labequipment import shaker
import time

start =560
end = 500
rate = 0.2

s = shaker.Shaker()
for n in range(30):
    s.ramp(start, end, rate, stop_at_end=False, record=False)
    s.init_duty(end)
    time.sleep(5)
    s.init_duty(end)
time.sleep(5)
s.quit()