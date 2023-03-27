import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

V2a = 42.46

# time = pd.read_csv("/media/ppxjd3/Nathan Backup Data V1/ByDate/September2019/Long_time_at_800/time.txt")
# accel = np.loadtxt("/media/ppxjd3/Nathan Backup Data V1/ByDate/September2019/Long_time_at_800/accel.txt")

time = pd.read_csv("/media/ppxjd3/Nathan Backup Data V1/ByDate/September2019/Final_setup_initial_temp/time.txt")
accel = np.loadtxt("/media/ppxjd3/Nathan Backup Data V1/ByDate/September2019/Final_setup_initial_temp/accel.txt")

# start_time = datetime.datetime.fromisoformat(time.DateTime.values[0])
time['DateTime'] = pd.to_datetime(time['DateTime'])

start_time = time.DateTime.values[0]

time['Diff'] = time['DateTime'] - start_time

diffs = time.Diff.values

hours = diffs.astype('float64')/1e9/3600

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(hours, accel)
ax2.plot(hours, accel*V2a/9.81)
ax1.set_xlabel('Hours')
ax1.set_ylabel('Voltage (V)')
ax2.set_ylabel('Dimensionless Acceleration $\Gamma$')
# plt.savefig('stable_800.jpg')
plt.show()