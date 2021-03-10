import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from labvision import images
import pandas as pd
from math import sqrt



### CONSTANT ####
FONTSIZE = 12
XMIN = 100
XMAX = 700
FIGX = 6
FIGY = 6
LINEWIDTH = 2
PANELS = 3

liquid_file = "/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/150121_liquid_610_45_down.txt"
solid_file = "/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/150121_solid_610_45_down.txt"
solid_dir = "/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/150121_solid_610_45_down"
solid_im = "/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/150121_solid_610_45_down/5.png"


def get_elapsed_time(dir_):
    ims = os.listdir(dir_)
    nums = [int(s.split('.')[0]) for s in ims if s.endswith('.png')]
    last_im = pathlib.Path(f"{dir_}/{max(nums)}.png")
    first_im = pathlib.Path(f"{dir_}/{min(nums)}.png")
    return last_im.stat().st_mtime - first_im.stat().st_mtime

def get_pix_2_mm(file):
    im = images.read_img(file)
    # crop_result = images.crop_polygon(im)
    # p = crop_result.points
    # pixel_distance = sqrt((p[1, 0]-p[0, 0])**2 + (p[1, 1]-p[0, 1])**2)
    pixel_distance = 823
    mm_distance = 215
    print(f"pixel distance: {pixel_distance}")
    return mm_distance/pixel_distance


def hist_border(n, b):
    x = np.repeat(b, 2)
    y = np.repeat(n, 2)
    y = np.insert(y, 0, 0)
    y = np.insert(y, len(y), 0)
    return x, y

def get_dataframe():
    return pd.DataFrame.from_dict(
        {'Time (s)': np.linspace(0, elapsed_time, len(liquid[::2])),
         'L1': liquid[::2],
         'L2': liquid[1::2],
         'S1': solid[::2],
         'S2': solid[1::2]}
    ).set_index('Time (s)')


liquid = np.loadtxt(liquid_file)
solid = np.loadtxt(solid_file)
lx1, lx2 = liquid[::2], liquid[1::2]
sx1, sx2 = solid[::2], solid[1::2]

elapsed_time = get_elapsed_time(solid_dir)

pix2mm = get_pix_2_mm(solid_im)




solid = solid[:len(liquid)]

fig = plt.figure(figsize=(FIGX, FIGY), linewidth=LINEWIDTH, tight_layout=True)
ax1, ax_im, ax2 = fig.subplots(PANELS, 1, sharex=True, gridspec_kw = {'hspace': 0})
fig.subplots_adjust(hspace=0)

ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE, direction='in')
ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE, direction='in')

ax1.set_xlim(XMIN, XMAX)


bins = np.arange(XMIN, XMAX)
solid_n, solid_b, _ = ax1.hist(solid, bins=bins, alpha=0.5, label='solid', color='b', density=True)
ax1.plot(*hist_border(solid_n, solid_b), 'b')
liquid_n, liquid_b, _ = ax1.hist(liquid, bins=bins, alpha=0.5, label='liquid', color='r', density=True)
ax1.plot(*hist_border(liquid_n, liquid_b), 'r')
ax1.set_ylabel('Probability density', fontsize=FONTSIZE)
ax1.legend(fontsize=FONTSIZE)


time_data = get_dataframe()

rolling = time_data.rolling(window=1).mean()
ax2.plot(rolling.L1, rolling.index/3600, 'r', alpha=0.3)
ax2.plot(rolling.L2, rolling.index/3600, 'r', alpha=0.3)
ax2.plot(rolling.S1, rolling.index/3600, 'b', alpha=0.3)
ax2.plot(rolling.S2, rolling.index/3600, 'b', alpha=0.3)

rolling = time_data.rolling(window=600).mean()
ax2.plot(rolling.L1, rolling.index/3600, 'r', alpha=1)
ax2.plot(rolling.L2, rolling.index/3600, 'r', alpha=1)
ax2.plot(rolling.S1, rolling.index/3600, 'b', alpha=1)
ax2.plot(rolling.S2, rolling.index/3600, 'b', alpha=1)

ax2.set_xlabel('Position (mm)', fontsize=FONTSIZE)
ax2.set_ylabel('Time (hrs)', fontsize=FONTSIZE)


display_frame = images.load("/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Sample.png")
ax_im.imshow(display_frame[:, :, ::-1])

from matplotlib import ticker
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%d') % (x * pix2mm)))
ax_im.set_yticks([])
# ax_im.set_xticks([])

def vlines(ax, arr, color):
    ax.axvline(arr.mean(), linestyle='--', color=color)
    ax.axvline(arr.mean()-(arr.std()/sqrt(len(arr))), linestyle='--', color=color, alpha=0.5)
    ax.axvline(arr.mean()+(arr.std()/sqrt(len(arr))), linestyle='--', color=color, alpha=0.5)

# Add vertical lines to show the mean positions
vlines(ax1, lx1, 'r')
vlines(ax1, lx2, 'r')
vlines(ax1, sx1, 'b')
vlines(ax1, sx2, 'b')

# ax1.axvline(lx2.mean(), linestyle='--', color='r')
# ax1.axvline(sx1.mean(), linestyle='--', color='b')
# ax1.axvline(sx2.mean(), linestyle='--', color='b')

fig, ax = plt.subplots()
sep_bins = np.arange(200, 600, 5)
# sep_bins = 100
ax.hist(lx2-lx1, bins=sep_bins, alpha=0.5, color='r', density=True, label='liquid')
ax.hist(sx2-sx1, bins=sep_bins, alpha=0.5, color='b', density=True, label='solid')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%d') % (x * pix2mm)))
ax.set_xlabel('Separation (mm)')
ax.legend()
plt.show()