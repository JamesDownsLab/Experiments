import pandas as pd
import numpy as np
import filehandling
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from labvision import images
from ball_bearing_duty import duty_to_dimensionless_acceleration

matplotlib.rcParams.update({'font.size': 22})

directory = "/media/data/Data/BallBearing/HIPS/IslandExperiments"

Ls = ['1,87', '1,89', '1,91', '1,93', '1,96', '1,97', '1,98']

fig = plt.figure(figsize=(13.5, 21.4))
gridspec = GridSpec(len(Ls), 4, width_ratios=[6, 6, 6, 1])
cmap = matplotlib.cm.hsv
norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)

i = 0

def set_axis_props(ax):
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])

for L in Ls:
    j = 0
    L_float = float(L.replace(',', '.'))
    files = filehandling.get_directory_filenames(f"{directory}/{L}mmRepeats/*.hdf5")
    for file in files[-3:]:
        data = pd.read_hdf(file)
        p_rad = 8
        frame = data.loc[0]
        im = np.ones((2000, 2300, 3), dtype=np.uint8)*255
        x, y, c = frame.x, frame.y, cmap(norm(np.angle(frame.hexatic_order)))
        for xi, yi, ci in zip(x, y, c):
            im = images.draw_circle(im, xi, yi, p_rad, ci*255, thickness=-1)
        if j == 0:
            N = (2.00-L_float+0.57)/(2.00-L_float)
            R_circ = (N-1) * 2 * p_rad
            im = images.draw_circle(im, np.mean(x), np.mean(y), R_circ, color=images.PINK, thickness=10)
        ax = fig.add_subplot(gridspec[i, j])
        set_axis_props(ax)
        ax.imshow(im)
        # ax.scatter(frame.x, frame.y, color=cmap(norm(np.angle(frame.hexatic_order))))
        # ax.quiver(frame.x, frame.y,
        #           np.real(frame.hexatic_order), np.imag(frame.hexatic_order),
        #           color=cmap(norm(np.angle(data.hexatic_order))))
        if j == 0:
            ax.set_ylabel(L.replace(',', '.')+'mm')
        j += 1

    i += 1

cax = fig.add_subplot(gridspec[:, 3])
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                                orientation='vertical')

fig.tight_layout()
plt.savefig(f"{directory}/overview.png", dpi=600)