import pandas as pd
import numpy as np
import filehandling
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from ball_bearing_duty import duty_to_dimensionless_acceleration

matplotlib.rcParams.update({'font.size': 22})

directory = "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,31mm"

L = directory[-6:]

densities = ['65%', '75%', '85%']

duties = [520, 500, 490, 480]

def get_data_and_plot_quiver(duty, density, ax, show_title=False, show_ylabel=False):
    acc = duty_to_dimensionless_acceleration(duty)
    print(duty, acc)
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    # ax.yaxis.set_visible(False)
    if show_title:
        ax.set_title(density)
    if show_ylabel:
        ax.set_ylabel(f"{acc:.2f}")
    data = pd.read_hdf(f"{directory}/{density}/{duty}.hdf5")
    data = data.loc[0]
    ax.quiver(data.x, data.y, np.real(data.hexatic_order), np.imag(data.hexatic_order),
              color=cmap(norm(np.angle(data.hexatic_order.values))))

cmap = matplotlib.cm.hsv
norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)


fig = plt.figure(figsize=(13.5, 21.4))
gridspec = GridSpec(5, 4, width_ratios=[6, 6, 6, 1])

for density_i in range(3):
    for duty_i in range(4):
        ylabel = density_i == 0
        title = duty_i == 0
        ax = fig.add_subplot(gridspec[duty_i, density_i])
        get_data_and_plot_quiver(duties[duty_i], densities[density_i], ax,
                                 show_title=title, show_ylabel=ylabel)


cax = fig.add_subplot(gridspec[:4, 3])
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                                orientation='vertical')

order_ax = fig.add_subplot(gridspec[4, :])
for density in densities:
    all_duties = np.loadtxt(f"{directory}/{density}_duty.txt")
    accel = [duty_to_dimensionless_acceleration(d) for d in all_duties]
    order = np.loadtxt(f"{directory}/{density}_order_frame.txt")
    error = np.loadtxt(f"{directory}/{density}_order_frame_err.txt")
    order_ax.errorbar(accel, order, yerr=error, label=density)
for duty in duties:
    order_ax.axvline(duty_to_dimensionless_acceleration(duty), linestyle='--')
order_ax.legend()
order_ax.set_xlabel('$\Gamma$')
order_ax.set_ylabel(r'$|\bar{\psi_6}|$')

fig.text(0.05, 0.58, '$\Gamma$', fontsize=28)
fig.text(0.45, 0.90, 'Density', fontsize=28)


plt.savefig(f"{directory}/{L}_overview.png", dpi=600)


