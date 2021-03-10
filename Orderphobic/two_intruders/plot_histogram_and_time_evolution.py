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


# solid = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/FoamPlaIntruders/Logging/081220_x_solid_wide_gap_570.txt")
# liquid = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/FoamPlaIntruders/Logging/091220_x_liquid_wide_gap_570.txt")
save_dir = "/media/data/Data/Orderphobic/TwoIntruders/FoamPlaIntruders/Logging/08-09_logs"
lx1 = np.loadtxt(f"{save_dir}/lx1.txt")
lx2 = np.loadtxt(f"{save_dir}/lx2.txt")
sx1 = np.loadtxt(f"{save_dir}/sx1.txt")
sx2 = np.loadtxt(f"{save_dir}/sx2.txt")
liquid = np.insert(lx2, np.arange(len(lx1)), lx1)
solid = np.insert(sx2, np.arange(len(sx1)), sx1)

elapsed_time = get_elapsed_time("/media/data/Data/Orderphobic/TwoIntruders/FoamPlaIntruders/Logging/091220_x_liquid_wide_gap_570")

pix2mm = get_pix_2_mm("/media/data/Data/Orderphobic/TwoIntruders/FoamPlaIntruders/Logging/091220_x_liquid_wide_gap_570/36000.png")




solid = solid[:len(liquid)]

fig = plt.figure(figsize=(FIGX, FIGY), linewidth=LINEWIDTH, tight_layout=True)
ax1, ax_im, ax2 = fig.subplots(3, 1, sharex=True, gridspec_kw = {'hspace': 0})
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


display_frame = images.load("/media/data/Data/Orderphobic/TwoIntruders/FoamPlaIntruders/Logging/08-09_logs/sample_im.png")
ax_im.imshow(display_frame[:, :, ::-1])

from matplotlib import ticker
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%d') % (x * pix2mm)))
ax_im.set_yticks([])
# ax_im.set_xticks([])



# Add vertical lines to show the mean positions
ax1.axvline(lx1.mean(), linestyle='--', color='r')
ax1.axvline(lx2.mean(), linestyle='--', color='r')
ax1.axvline(sx1.mean(), linestyle='--', color='b')
ax1.axvline(sx2.mean(), linestyle='--', color='b')


plt.show()