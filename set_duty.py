import sys
from labequipment import shaker

duty = sys.argv[1]
duty = int(duty)

s = shaker.Shaker()
s.init_duty(duty)

