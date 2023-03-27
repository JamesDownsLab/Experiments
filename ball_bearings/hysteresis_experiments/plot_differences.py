import filehandling
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

rate = '0,4'
L = '2,10'
suf = '_b'

down_files = filehandling.get_directory_filenames(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/{rate}_*_down_duty_mean_corr.hdf5")
up_files = filehandling.get_directory_filenames(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/{rate}_*_up_duty_mean_corr.hdf5")

down_orders = defaultdict(list)
for file in down_files:
    data = pd.read_hdf(file)
    duties = data.index.values
    orders = data.values
    for duty, order in zip(duties, orders):
        down_orders[duty].append(order)

up_orders = defaultdict(list)
for file in up_files:
    data = pd.read_hdf(file)
    duties = data.index.values
    orders = data.values
    for duty, order in zip(duties, orders):
        up_orders[duty].append(order)

duties = []
mean_diffs = []
for duty in down_orders.keys():
    down_order = down_orders[duty]
    up_order = up_orders[duty]
    diffs = np.array(up_order) - np.array(down_order)
    duties.append(duty)
    mean_diffs.append(np.mean(diffs))

np.savetxt(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/diff_plot_data/{rate}_duty.txt", duties)
np.savetxt(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/diff_plot_data/{rate}_diff.txt", mean_diffs)

plt.plot(duties, mean_diffs)
plt.xlabel('Duty')
plt.ylabel('$\Delta \psi_6$')
plt.title(rate)
plt.savefig(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/diff_plot_data/{rate}.png")