import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import filehandling

# filename = "/media/NAS/share/James/SimulationDataSep22/87_3,5e-4.csv"

# files = filehandling.get_directory_filenames("/media/NAS/share/James/SimulationDataSep22/*.csv")

files = [
    "/media/NAS/share/James/SimulationDataSep22/95_4e-4.csv",

]

def save_sampled_hoz_data(filename):
    data = pd.read_csv(filename)
    # Add horizontal velocity
    data['v_hoz'] = np.sqrt(data['vx'] ** 2 + data['vy'] ** 2)

    frame_numbers = np.unique(data['frame'].values)
    times = np.unique(data['time'].values)

    frame_interval = frame_numbers[1] - frame_numbers[0]
    dt = (times[1] - times[0]) / frame_interval
    sample_time = times[1] - times[0]

    save_data = data[['frame', 'v_hoz']].copy()

    for step in [1, 5, 10, 50, 100, 500, 1000]:
        print(f"Calculating diff for step = {step}")
        datadiff = data.groupby('particle').diff(step)
        save_data[f"v_hoz_{round(1 / (step * sample_time))}_Hz"] = np.sqrt(
            datadiff.x ** 2 + datadiff.y ** 2) / datadiff.time


    print(save_data.head())
    save_data.to_csv(filename[:-4] + '_savedata.hdf5')

for filename in files:
    save_sampled_hoz_data(filename)