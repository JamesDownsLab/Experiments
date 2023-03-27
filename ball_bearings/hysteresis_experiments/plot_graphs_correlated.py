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

down_orders = {key: np.mean(values) for key, values in down_orders.items()}

up_orders = defaultdict(list)
for file in up_files:
    data = pd.read_hdf(file)
    duties = data.index.values
    orders = data.values
    for duty, order in zip(duties, orders):
        up_orders[duty].append(order)

up_orders = {key: np.mean(values) for key, values in up_orders.items()}

np.savetxt(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/plot_data/{rate}_down_duty_corr.txt", list(down_orders.keys()))
np.savetxt(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/plot_data/{rate}_up_duty_corr.txt", list(up_orders.keys()))
np.savetxt(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/plot_data/{rate}_down_order_corr.txt", list(down_orders.values()))
np.savetxt(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/plot_data/{rate}_up_order_corr.txt", list(up_orders.values()))

plt.figure()
plt.plot(down_orders.keys(), down_orders.values(), 'c', label='cooling')
plt.plot(up_orders.keys(), up_orders.values(), 'r', label='heating')
plt.xlabel('Duty')
plt.ylabel('$|\psi_6|$')
plt.legend()
plt.savefig(f"/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/{L}mm{suf}/plot_data/{rate}_corr.png")