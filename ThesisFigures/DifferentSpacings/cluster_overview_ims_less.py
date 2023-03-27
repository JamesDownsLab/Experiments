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

Ls = ['1,87', '1,89', '1,91', '1,93', '1,96', '1,98']

fig = plt.figure(figsize=(13.5, 21.4))
gridspec = GridSpec(3, 3, width_ratios=[6, 6, 1])
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
    for file in files[-1:]:
        data = pd.read_hdf(file)
        p_rad = 8
        frame = data.loc[0]
        im = np.ones((2000, 2300, 3), dtype=np.uint8)*255
        x, y, c = frame.x, frame.y, cmap(norm(np.angle(frame.hexatic_order)))
        for xi, yi, ci in zip(x, y, c):
            im = images.draw_circle(im, xi, yi, p_rad, ci*255, thickness=-1)
        N = (2.00-L_float+0.57)/(2.00-L_float)
        R_circ = (N-1) * 2 * p_rad
        im = images.draw_circle(im, np.mean(x), np.mean(y), R_circ, color=images.PINK, thickness=10)
        ax = fig.add_subplot(gridspec[i//2, i%2])
        set_axis_props(ax)
        ax.imshow(im)
        ax.set_title(L.replace(',', '.') + ' mm')
        # ax.scatter(frame.x, frame.y, color=cmap(norm(np.angle(frame.hexatic_order))))
        # ax.quiver(frame.x, frame.y,
        #           np.real(frame.hexatic_order), np.imag(frame.hexatic_order),
        #           color=cmap(norm(np.angle(data.hexatic_order))))

    i += 1

cax = fig.add_subplot(gridspec[:, 2])
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                                orientation='vertical')

fig.tight_layout()
plt.savefig(f"{directory}/overview_less.png", dpi=600)