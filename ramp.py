import sys
from labequipment import shaker

start, end, rate = sys.argv[1:4]
start = int(start)
end = int(end)
rate = float(rate)
record = "--record" in sys.argv
stop_at_end = "--stop_at_end" in sys.argv

s = shaker.Shaker()
s.ramp(start, end, rate, stop_at_end=stop_at_end, record=record)
s.quit()