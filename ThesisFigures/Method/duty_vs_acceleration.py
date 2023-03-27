import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

V2a = 42.46

data = pd.read_csv("/media/data/Data/GeneralSystem/DutyAccelerationMaps/FullBlueTray/AccelerationData.csv",
                   names=['Duty', 'RMS', 'RMS_error'], header=0)

fig, ax1 = plt.subplots()
accel = data.RMS/1000 * V2a
plt.plot(data.Duty/10, accel)
plt.axvline(58, ls='--')
plt.axvline(68, ls='--')
ax1.set_xlabel('Duty Cycle, %')
ax1.set_ylabel('Acceleration, m/s$^2$')

ax2 = ax1.twinx()
ax2.plot(data.Duty/10, accel/9.81)
ax2.set_ylabel('Dimensionless Acceleration $\Gamma$')
plt.savefig('DutyAcceleration.jpg')

