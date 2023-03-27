import pandas as pd
import numpy as np
import filehandling
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from ball_bearing_duty import duty_to_dimensionless_acceleration

matplotlib.rcParams.update({'font.size': 22})

directory = "/media/data/Data/BallBearing/HIPS/IslandExperiments"

Ls = ['1,87', '1,89', '1,91', '1,93', '1,96', '1,97', '1,98']

fig = plt.figure(figsize=(13.5, 21.4))
gridspec = GridSpec(len(Ls), 4, width_ratios=[6, 6, 6, 1])
cmap = matplotlib.cm.hsv
norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)

x = 0

def set_axis_props(ax):
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])

for L in Ls:
    y = 0
    L_float = float(L.replace(',', '.'))
    files = filehandling.get_directory_filenames(f"{directory}/{L}mmRepeats/*.hdf5")
    for file in files[-3:]:
        data = pd.read_hdf(file)
        frame = data.loc[0]
        ax = fig.add_subplot(gridspec[x, y])
        set_axis_props(ax)
        ax.scatter(frame.x, frame.y, color=cmap(norm(np.angle(frame.hexatic_order))))
        # ax.quiver(frame.x, frame.y,
        #           np.real(frame.hexatic_order), np.imag(frame.hexatic_order),
        #           color=cmap(norm(np.angle(data.hexatic_order))))
        if y == 0:
            ax.set_ylabel(L.replace(',', '.')+'mm')
        y += 1

    x += 1

fig.tight_layout()
plt.savefig(f"{directory}/overview.png", dpi=600)