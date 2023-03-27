import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm, colorbar
import pandas as pd
from order_parameters import add_translational_order
from scipy import spatial

def get_diameter(data):
    tree = spatial.KDTree(data[['x', 'y']].values)
    dists, indices = tree.query(data[['x', 'y']].values, 7)
    dists = dists[:, 1:]
    dists = np.ndarray.flatten(dists)
    return np.mean(dists)

def remove_all_axis_marks(ax):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False
    )

def plot_horder_torder_horizontal(data, fname):
    fig, (ax1, ax2, cax) = plt.subplots(1, 3, squeeze=True, figsize=(5.5, 2.25), gridspec_kw={'width_ratios': [10, 10, 1]})
    remove_all_axis_marks(ax1)
    remove_all_axis_marks(ax2)
    ax1.set_aspect('equal')

    ax2.set_aspect('equal')

    cmap = cm.hsv
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)

    diameter = get_diameter(data)

    for x, y, h, t in zip(data.x, data.y, data.hexatic_order, data.translational_order):
        c = plt.Circle((x, y), diameter/4, color=cmap(norm(np.angle(h))))
        ax1.add_artist(c)
        c = plt.Circle((x, y), diameter/4, color=cmap(norm(np.angle(t))))
        ax2.add_artist(c)
    ax1.set_xlim([data.x.min(), data.x.max()])
    ax1.set_ylim([data.y.min(), data.y.max()])
    ax2.set_xlim([data.x.min(), data.x.max()])
    ax2.set_ylim([data.y.min(), data.y.max()])

    ax1.set_title('$\psi_6$')
    ax2.set_title('$\psi_T$')

    cbar = colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical', ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])

    fig.savefig(fname+'.pdf', format='pdf', bbox_inches='tight')

def get_data_and_plot(fname, savename, save_directory):
    data = pd.read_hdf(fname)
    data = data.loc[0].copy()
    data, _ = add_translational_order(data, refine=True)
    plot_horder_torder_horizontal(data, save_directory+'/'+savename)

if __name__ == '__main__':
    save_directory = "/media/data/Data/ThesisFigures/5) Different Spacings/Overview 2mm"
    files = [
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,00mm/85%/500.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,00mm/85%/520.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,00mm/85%/540.hdf5",
    ]

    names = [
        '2,00_85%_500',
        '2,00_85%_520',
        '2,00_85%_540'
    ]
    for f, n in zip(files, names):
        get_data_and_plot(f, n, save_directory)