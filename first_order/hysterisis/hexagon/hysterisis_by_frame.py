from particletracking import dataframes
import filehandling
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import numpy as np
import duty


PARAMETER = 'order'
labels = {'up': 'heating', 'down': 'cooling'}
COLORS = {'up': 'r', 'down': 'c'}

direc = "/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense"
save_direc = f"/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense/Results_hexed_{PARAMETER}_frame"
data_direc = "/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense/HexagonFigures/data"
files = filehandling.get_directory_filenames(direc + '/*.hdf5')


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
            dim_acc = duty.d2G(mean.duty_interpolated)
            ax[i].errorbar(dim_acc, mean[PARAMETER], yerr=std[PARAMETER], fmt=COLORS[direction], label=labels[direction], errorevery=500)
            ax[i].set_ylim([0.6, 0.9])
            ax[i].set_xlim([duty.d2G(620),
                            duty.d2G(680)])
            ax[i].text(duty.d2G(625), 0.7, f'rate = {rate}')
            # np.savetxt(f'{data_direc}/{PARAMETER}_frame_x_{direction}_{rate}.txt', dim_acc)
            # np.savetxt(f'{data_direc}/{PARAMETER}_frame_y_{direction}_{rate}.txt', mean[PARAMETER])
            # np.savetxt(f'{data_direc}/{PARAMETER}_frame_yerr_{direction}_{rate}.txt', std[PARAMETER])
            savedata = pd.DataFrame({'x': dim_acc, 'y': mean[PARAMETER], 'y+-': std[PARAMETER]})
            savedata.to_csv(f'{data_direc}/frame_{PARAMETER}_{direction}_{rate}.csv')


    ax[0].legend()
    ax[0].set_title('By frame')
    ax[1].set_ylabel(PARAMETER)
    ax[2].set_xlabel('$\Gamma$')
    plt.tight_layout()

def plot_all(results):
    fig, ax = plt.subplots(3, 1, sharex='all')
    for i, (rate, result) in enumerate(results.items()):
        colors = ['k', 'r', 'b', 'y', 'g']
        for direction, trials in result.items():
            for j in range(len(trials)):
                trial = trials[j]
                trial['duty_interpolated'] = get_duty_from_frame(trial.index, direction, 700, 600)
                dim_acc = duty.d2G(trial['duty_interpolated'])
                ax[i].plot(dim_acc, trial[PARAMETER], color=colors[j], label=j)
        ax[i].set_ylim([0.6, 0.9])
        ax[i].set_xlim([duty.d2G(620),
                        duty.d2G(680)])
        ax[i].text(duty.d2G(625), 0.7, f'rate = {rate}')

    ax[0].set_title('By frame')
    ax[1].set_ylabel(PARAMETER)
    ax[2].set_xlabel('$\Gamma$')
    plt.tight_layout()

plot_mean(results)
plot_all(results)
plt.show()



plt.show()