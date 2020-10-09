from particletracking import dataframes
import filehandling
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import numpy as np
import duty

PARAMETER = 'order'

direc = "/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense"
save_direc = f"/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense/Results_hexed_{PARAMETER}_duty"
files = filehandling.get_directory_filenames(direc + '/*.hdf5')
data_direc = "/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense/HexagonFigures/data"
labels = {'up': 'heating', 'down': 'cooling'}
COLORS = {'up': 'r', 'down': 'c'}

def filter_box(df, xmin, xmax, ymin, ymax):
    df = df.loc[(df.x > xmin) * (df.x < xmax) * (df.y > ymin) * (df.y < ymax)]
    return df

def filter_hexagon(df, corners, factor):
    center = np.mean(corners, axis=0)
    vectors = corners - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    lengths = np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)
    new_vectors = np.array([np.cos(angles), np.sin(angles)]).T
    new_vectors *= lengths[:, np.newaxis] * factor
    new_corners = center + new_vectors
    path = mplpath.Path(new_corners)
    include = path.contains_points(df[['x', 'y']].values)
    return df[include]


def calculate_and_save_data(files):
    for file in tqdm(files):
        filename = os.path.splitext(os.path.split(file)[1])[0]
        rate, direction, trial = filename.split('_')
        trial = int(trial)
        data = dataframes.DataStore(file)
        df = data.df
        df.Duty = df.Duty.round()
        # df = filter_box(df, 750, 1250, 750, 1250)
        df = filter_hexagon(df, data.metadata['boundary'], 0.8)

        result = df.groupby('Duty')[PARAMETER].mean()
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

def plot_mean(results):
    fig, ax = plt.subplots(3, 1, sharex='all')
    for i, (rate, result) in enumerate(results.items()):
        for direction, trials in result.items():
            # for j in range(len(trials)):
            #     trials[j] = trials[j].set_index('Duty')
            data = pd.concat(trials)
            mean = data.groupby('Duty').mean()
            std = data.groupby('Duty').std()
            dim_acc = duty.d2G(mean.index)
            ax[i].errorbar(dim_acc, mean[PARAMETER], yerr=std[PARAMETER], fmt=COLORS[direction], label=labels[direction])
            ax[i].set_ylim([0.6, 0.9])
            ax[i].set_xlim([duty.d2G(620),
                            duty.d2G(680)])
            ax[i].text(duty.d2G(625), 0.7, f'rate = {rate}')
            # np.savetxt(f'{data_direc}/{PARAMETER}_duty_x_{direction}_{rate}.txt',
            #            dim_acc)
            # np.savetxt(f'{data_direc}/{PARAMETER}_duty_y_{direction}_{rate}.txt',
            #            mean[PARAMETER])
            # np.savetxt(f'{data_direc}/{PARAMETER}_duty_yerr_{direction}_{rate}.txt',
            #            std[PARAMETER])
            savedata = pd.DataFrame(
                {'x': dim_acc, 'y': mean[PARAMETER].values, '+-': std[PARAMETER].values})
            savedata.to_csv(f'{data_direc}/duty_{PARAMETER}_{direction}_{rate}.csv')
    # for i in range(2):
    #     ax[i].set_xticks([])
    ax[0].set_title('By Duty')
    ax[0].legend()
    ax[1].set_ylabel(PARAMETER)
    ax[2].set_xlabel('$\Gamma$')
    plt.tight_layout()


plot_mean(results)


def plot_all(results):
    fig, ax = plt.subplots(3, 1, sharex='all')
    for i, (rate, result) in enumerate(results.items()):
        for direction, trials in result.items():
            colors = ['r', 'g', 'y', 'b', 'k']
            for j in range(len(trials)):
                trials[j] = trials[j].groupby('Duty').mean()
                acc = duty.d2G(trials[j].index)
                ax[i].plot(acc, trials[j][PARAMETER], color=colors[j], label=j)
        # ax[i].set_title(rate)
        ax[i].set_ylim([0.6, 0.9])
        ax[i].set_xlim([duty.d2G(620), duty.d2G(680)])
        ax[i].text(duty.d2G(625), 0.7, f'rate = {rate}')

    ax[0].set_title('By Duty')
    ax[1].set_ylabel(PARAMETER)
    # ax[0].legend()
    ax[2].set_xlabel('$\Gamma$')
    plt.tight_layout()
plot_all(results)

plt.show()