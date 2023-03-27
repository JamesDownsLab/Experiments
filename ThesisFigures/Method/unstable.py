import numpy as np
import matplotlib.pyplot as plt

t = np.loadtxt("/media/ppxjd3/Nathan Backup Data V1/ByDate/September2019/Accel_and_temp_with_radiator2/time.txt")
v = np.loadtxt("/media/ppxjd3/Nathan Backup Data V1/ByDate/September2019/Accel_and_temp_with_radiator2/accel.txt")

plt.plot(t, v)
plt.show()