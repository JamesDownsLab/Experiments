from particletracking import dataframes
import filehandling
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PARAMETER = 'order'


direc = "/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense"
save_direc = f"/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense/Results_boxed_{PARAMETER}_frame"
files = filehandling.get_directory_filenames(direc + '/*.hdf5')


def filter_box(df, xmin, xmax, ymin, ymax):
    df = df.loc[(df.x > xmin) * (df.x < xmax) * (df.y > ymin) * (df.y < ymax)]
    return df

def calculate_and_save_data(files):
    for file in tqdm(files):
        filename = os.path.splitext(os.path.split(file)[1])[0]
        rate, direction, trial = filename.split('_')
        trial = int(trial)
        df = dataframes.DataStore(file).df
        df.Duty = df.Duty.round()
        df = filter_box(df, 750, 1250, 750, 1250)
        result = df.groupby('frame')[PARAMETER].mean()
        result.to_csv(f'{save_direc}/{rate}_{direction}_{trial}')


# calculate_and_save_data(files)

data_files = filehandling.get_directory_filenames(save_direc+'/*')

def populate_results(data_files):
    results = {}
    for file in tqdm(data_files, 'populating dictionary'):
        data = pd.read_csv(file, engine='python')
        filename = os.path.split(file)[1]
        rate, direction, trial = filename.split('_')
        if rate not in results.keys():
            results[rate] = {'up': [], 'down': []}
        results[rate][direction].append(data)
    return results

results = populate_results(data_files)

def get_duty_from_frame(frame_col, direction, top, bottom):
    if direction is 'down':
        return np.linspace(top, bottom, len(frame_col))
    else:
        return np.linspace(bottom, top, len(frame_col))

def plot_mean(results):
    fig, ax = plt.subplots(3, 1, sharex='all')
    for i, (rate, result) in enumerate(results.items()):
        for direction, trials in result.items():
            data = pd.concat(trials)
            mean = data.groupby('frame').mean()
            std = data.groupby('frame').std()
            mean['duty_interpolated'] = get_duty_from_frame(mean.index, direction, 700, 600)
            ax[i].errorbar(mean.duty_interpolated, mean[PARAMETER], yerr=std[PARAMETER], label=direction, errorevery=500)
            ax[i].set_ylim([0.6, 0.9])
            ax[i].set_xlim([620, 680])
            ax[i].text(625, 0.7, f'rate = {rate}')

    # ax[0].legend()
    ax[0].set_title('By frame')
    ax[1].set_ylabel(PARAMETER)
    ax[2].set_xlabel('Duty')
    plt.tight_layout()

def plot_all(results):
    fig, ax = plt.subplots(3, 1, sharex='all')
    for i, (rate, result) in enumerate(results.items()):
        colors = ['k', 'r', 'b', 'y', 'g']
        for direction, trials in result.items():
            for j in range(len(trials)):
                trial = trials[j]
                trial['duty_interpolated'] = get_duty_from_frame(trial.index, direction, 700, 600)
                ax[i].plot(trial['duty_interpolated'], trial[PARAMETER], color=colors[j], label=j)
        ax[i].set_ylim([0.6, 0.9])
        ax[i].set_xlim([620, 680])
        ax[i].text(625, 0.7, f'rate = {rate}')

    ax[0].set_title('By frame')
    ax[1].set_ylabel(PARAMETER)
    ax[2].set_xlabel('Duty')
    plt.tight_layout()

plot_mean(results)
plot_all(results)
plt.show()



plt.show()