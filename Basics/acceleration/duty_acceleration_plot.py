import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("/media/data/Data/August2020/DutyAcceleration/FullBlueTray/AccelerationData.csv",
                   names=['Duty', 'RMS', 'RMS_error'], header=0)

plt.figure()
plt.errorbar(data.Duty/10, data.RMS, yerr=data.RMS_error, fmt='x')
plt.xlabel('Duty (%)')
plt.ylabel('RMS voltage (mV)')
plt.show()