import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import filehandling
from tqdm import tqdm
from os.path import splitext
import time

files = filehandling.get_directory_filenames("/media/NAS/share/James/SimulationDataSep22/*.csv")

steps = [
    10,
    25,
    50,
    100,
    125,
    150,
    200,
    250,
    300,
    500,
    1000
]

def save_sampled_hoz_data(filename):
    filecore = splitext(filename)[0]
    data = pd.read_csv(filename)
    # Add horizontal velocity
    data['v_hoz'] = np.sqrt(data['vx'] ** 2 + data['vy'] ** 2)

    frame_numbers = np.unique(data['frame'].values)
    times = np.unique(data['time'].values)

    frame_interval = frame_numbers[1] - frame_numbers[0]
    dt = (times[1] - times[0]) / frame_interval
    sample_time = times[1] - times[0]

    save_data = data[['frame', 'v_hoz']].copy()

    data = data[['x', 'y', 'particle', 'time']]

    for step in tqdm(steps, desc="Steps: ", position=1, leave=False):
        datadiff = data.groupby('particle').diff(step).abs()
        Lx = datadiff.x.max()
        Ly = datadiff.y.max()
        datadiff.x[datadiff.x > Lx/2] = Lx - datadiff.x
        datadiff.y[datadiff.y > Ly/2] = Ly - datadiff.y
        save_data[f"v_hoz_{round(1 / (step * sample_time))}_Hz"] = np.sqrt(
            datadiff.x ** 2 + datadiff.y ** 2) / datadiff.time

    group = save_data.groupby('frame')

    medians = group.median()
    medians.to_csv(filecore+'.medians')

    means = group.mean()
    means.to_csv(filecore+'.means')


for filename in tqdm(files, desc="Files: ", position=0):
    starttime = time.time()
    save_sampled_hoz_data(filename)
    end_time = time.time()
    print(f"Time elapsed: {end_time-starttime} s")