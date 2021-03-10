import matplotlib.pyplot as plt
import numpy as np
from labvision import camera, images
import os
import pathlib
import matplotlib as mpl

mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.linewidth'] = 2

solid = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/280121_solid_590_fat_clamps_2.txt")
liquid = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/280121_liquid_590_fat_clamps_less.txt")

solid_dir = "/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/280121_solid_590_fat_clamps_2"
liquid_dir = "/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/280121_liquid_590_fat_clamps_less"

solid = solid[:len(liquid)]

solid_ims = os.listdir(solid_dir)
print(solid_ims)
solid_nums = [int(s.split('.')[0]) for s in solid_ims if s.endswith('.png')]
last_im = pathlib.Path(f"{solid_dir}/{max(solid_nums)}.png")
first_im = pathlib.Path(f"{solid_dir}/{min(solid_nums)}.png")
elapsed_time = last_im.stat().st_mtime - first_im.stat().st_mtime

liquid = liquid[:len(solid)]

def hist_border(n, b):
    x = np.repeat(b, 2)
    y = np.repeat(n, 2)
    y = np.insert(y, 0, 0)
    y = np.insert(y, len(y), 0)
    return x, y

def plot_border(n, b, c):
    x, y = hist_border(n, b)
    plt.plot(x, y, c)

bins = np.arange(100, 700)
plt.subplot(2, 1, 1)
solid_n, solid_b, _ = plt.hist(solid, bins=bins, alpha=0.5, label='solid', color='r')
plot_border(solid_n, solid_b, 'r')
liquid_n, liquid_b, _ = plt.hist(liquid, bins=bins, alpha=0.5, label='liquid', color='b')
plot_border(liquid_n, liquid_b, 'b')
plt.legend()
plt.xlabel('Position (pixels)', labelpad=10)
plt.ylabel('Frequency', labelpad=10)



# liquid_distance = liquid[1::2] - liquid[::2]
# solid_distance = solid[1::2] - solid[::2]
#
# bins = np.arange(100, 600)
# plt.figure()
# s_n, s_b, _ = plt.hist(solid_distance, label='solid', bins=bins, alpha=0.5, color='r')
# plot_border(s_n, s_b, 'r')
#
# l_n, l_b, _ = plt.hist(liquid_distance, bins=bins, label='liquid', alpha=0.5, color='b')
# plot_border(l_n, l_b, 'b')
# plt.legend()
# plt.xlabel('Distance (pixels)')
# plt.ylabel('Frequency')



import pandas as pd
time_seconds = np.linspace(0, elapsed_time, len(liquid[::2]))
liquid_left = liquid[::2]
liquid_right = liquid[1::2]
solid_left = solid[::2]
solid_right = solid[1::2]

time_data = pd.DataFrame.from_dict(
    {'Time (s)': time_seconds,
     'L1': liquid_left,
     'L2': liquid_right,
     'S1': solid_left,
     'S2': solid_right}).set_index('Time (s)')

print(len(time_data))


def plot_rolling_average(time_data, window, alpha, label):
    rolling_average = time_data.rolling(window=window).mean()
    plt.plot(rolling_average.L1, rolling_average.index/3600, 'b', label=f'liquid {label}', alpha=alpha)
    plt.plot(rolling_average.L2, rolling_average.index/3600, 'b', alpha=alpha)
    plt.plot(rolling_average.S1, rolling_average.index/3600, 'r', label=f'solid {label}', alpha=alpha)
    plt.plot(rolling_average.S2, rolling_average.index/3600, 'r', alpha=alpha)
    plt.xlabel('Position (pixels)')
    plt.ylabel('Time (hrs)')
    plt.legend()

plt.subplot(2, 1, 2)
plot_rolling_average(time_data, 1, 0.3, 'raw')
plot_rolling_average(time_data, 273, 1, '5 minute')
plt.show()

