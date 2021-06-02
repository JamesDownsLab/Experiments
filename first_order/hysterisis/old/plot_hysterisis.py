from particletracking import statistics, dataframes
import filehandling
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

direc = "/media/data/Data/August2020/Hysterisis/RedTray2"
save_direc = "/media/data/Data/August2020/Hysterisis/RedTray2/Results_boxed_density_frame"
files = filehandling.get_directory_filenames(direc + '/*.hdf5')
#
# for file in files:
#     print(file)
#     filename = os.path.splitext(os.path.split(file)[1])[0]
#     rate, direction, trial = filename.split('_')
#     trial = int(trial)
#     df = dataframes.DataStore(file).df
#     if file == files[0]:
#         r = df.r.mean()
#     df = df.loc[(df.x > 750) * (df.x < 1250) * (df.y > 750) * (df.y < 1250)]
#     # df = df.loc[df.edge_distance > 5*r]
#     df.Duty = df.Duty.round()
#     # xmean = df.x.mean()
#     # df = df.loc[df.x > xmean]
#
#     result = df.groupby(df.index)['density'].mean()
#     result.to_csv(save_direc+'/{}_{}_{}'.format(rate, direction, trial))

data_files = filehandling.get_directory_filenames(save_direc+'/*')


results = {}
for file in data_files:
    data = pd.read_csv(file, engine='python')
    filename = os.path.split(file)[1]
    rate, direction, trial = filename.split('_')
    if rate not in results.keys():
        results[rate] = {'up': [], 'down': []}
    results[rate][direction].append(data)

# plt.figure()
# i = 0
# for rate, result in results.items():
#     print(rate)
#     i += 1
#     plt.subplot(2, 2, i)
#     # plt.title(rate)
#     # plt.figure()
#     plt.title(rate)
#     # colors = ['k', 'r', 'b', 'g', 'y']
#     ax1 = plt.gca()
#     # ax2 = ax1.twinx()
#     for direction, trials in result.items():
#         print(direction)
#         for j in range(len(trials)):
#             trials[j] = trials[j].set_index('Duty')
#         data = pd.concat(trials)
#         # data = trials].set_index('Duty')
#         mean = data.groupby(data.index).mean()
#         std = data.groupby(data.index).std()
#         count = data.groupby(data.index).count()
#         ax1.errorbar(mean.index[count.density==5], mean.density[count.density==5], yerr=std.density[count.density==5], label=direction)
#         # ax2.plot(count.index, count.density, 'r-')
#         # plt.plot(mean.index, mean.density, label=direction)
#     plt.legend()
# plt.show()

mean_fig, mean_axs = plt.subplots(2, 2)
all_fig, all_axs = plt.subplots(2, 2)
all_diff_fig, all_dif_axs = plt.subplots(2, 2)
mean_diff_fig, mean_diff_ax = plt.subplots()
for i, (rate, result) in enumerate(results.items()):
    x = 1 if i == 1 else 0
    y = 0 if i < 2 else 1
    mean_ax = mean_axs[x, y]
    all_ax = all_axs[x, y]
    all_diff_ax = all_dif_axs[x, y]
    mean_ax.set_title(rate)
    all_ax.set_title(rate)
    all_diff_ax.set_title(rate)
    colors = ['k', 'r', 'b', 'y', 'g']

    ups = result['up']
    downs = result['down']
    diffs = []
    N = 3 if rate == '0.2' else 5
    for k in range(N):
        up = ups[k]#.set_index('Duty')
        down = downs[k]#.set_index('Duty')
        diff = up-down
        diffs.append(diff)
        all_diff_ax.plot(diff.index, diff.density)
    mean_diff = pd.concat(diffs)
    mean_diff = mean_diff.groupby(mean_diff.index).mean()
    mean_diff_ax.plot(mean_diff.index, mean_diff.density, label=rate)


    for direction, trials in result.items():
        for j in range(len(trials)):
            trials[j] = trials[j]#.set_index('Duty')
            all_ax.plot(trials[j].index, trials[j].density, colors[j])

        data = pd.concat(trials)
        count = data.groupby(data.index).count().values.squeeze()
        mean = data.groupby(data.index).mean()#[count >= 3]
        mean['rolled'] = mean.density.rolling(100).mean()
        print(mean)
        std = data.groupby(data.index).std()#[count >= 3]
        # print(mean.head())
        if direction == 'up':
            print('up')
            mean.frame = mean.frame.values[::-1]
        # mean_ax.errorbar(mean.frame, mean.density, yerr=std.density, label=direction)
        mean_ax.plot(mean.frame, mean.rolled, label=direction)
    mean_ax.set_xlim([0, 50000])
    mean_ax.legend()
    all_ax.legend()
mean_diff_ax.legend()
plt.show()
