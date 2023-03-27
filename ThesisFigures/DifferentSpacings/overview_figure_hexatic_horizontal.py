import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm, colorbar
import pandas as pd
from order_parameters import add_translational_order
from scipy import spatial
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def plot(datasets, fname):
    fig = plt.figure(figsize=(7, 3))
    spec = fig.add_gridspec(1, len(datasets))

    cmap = cm.hsv
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    diameter = get_diameter(datasets[0])
    ymin = 1000
    ymax = 0
    xmin = 1000
    xmax = 0
    axs = []
    for i, data in enumerate(datasets):
        ax = fig.add_subplot(spec[0, i])
        axs.append(ax)
        ymin = ymin if ymin < data.y.min() else data.y.min()
        ymax = ymax if ymax > data.y.max() else data.y.max()
        xmin = xmin if xmin < data.x.min() else data.x.min()
        xmax = xmax if xmax > data.x.max() else data.x.max()
        # for x, y, h in zip(data.x, data.y, data.hexatic_order):
        #     # c = plt.Circle((x, y), diameter/2, color=cmap(norm(np.angle(h))))
        #     # ax.add_artist(c)
        h = data.hexatic_order.values
        ax.quiver(data.x, data.y, h.real, h.imag, color=cmap(norm(np.angle(h))))
        ax.set_aspect('equal', 'box')
        remove_all_axis_marks(ax)
    for ax in axs:
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    cax = inset_axes(axs[-1], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical', ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])

    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()
    fig.savefig(fname+'.pdf', format='pdf', bbox_inches='tight')
    # plt.show()



def get_data_and_plot(fnames, savename):
    datasets = []
    for fname in fnames:
        data = pd.read_hdf(fname)
        data = data.loc[0][['x', 'y', 'hexatic_order']]
        datasets.append(data)
    plot(datasets, savename)


if __name__ == '__main__':
    d = "/home/ppxjd3/Thesis/Thesis/Chapters/Different Spacings/Figures"
    # savename = f"{d}/2,00_85_hoz"
    # files = [
    #     "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,00mm/85%/500.hdf5",
    #     "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,00mm/85%/520.hdf5",
    #     "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,00mm/85%/540.hdf5",
    # ]
    # get_data_and_plot(files[::-1], savename)

    savename = f"{d}/2,15_85_hoz"
    files = [
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,15mm/85%/470.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,15mm/85%/480.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,15mm/85%/490.hdf5"
    ]
    get_data_and_plot(files[::-1], savename)

    savename = f"{d}/2,31_85_hoz"
    files = [
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,31mm/85%/510.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,31mm/85%/517.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,31mm/85%/520.hdf5"
    ]
    get_data_and_plot(files[::-1], savename)

    savename = f"{d}/2,42_80_hoz"
    files = [
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,42mm/80%/470.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,42mm/80%/480.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,42mm/80%/490.hdf5"
    ]
    get_data_and_plot(files[::-1], savename)

    savename = f"{d}/1,95_70_hoz"
    files = [
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/1,95mm/70%/540.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/1,95mm/70%/520.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/1,95mm/70%/500.hdf5"
    ]
    get_data_and_plot(files, savename)

    savename = f"{d}/1,91_85_hoz"
    files = [
        "/media/data/Data/BallBearing/HIPS/IslandExperiments/New_1,91mm_experiments/85%_15s/500.hdf5",
        "/media/data/Data/BallBearing/HIPS/IslandExperiments/New_1,91mm_experiments/85%_15s/485.hdf5",
        "/media/data/Data/BallBearing/HIPS/IslandExperiments/New_1,91mm_experiments/85%_15s/460.hdf5"
    ]
    get_data_and_plot(files, savename)

    savename = f"{d}/2,25_75_hoz"
    files = [
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,25mm/75%/490.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,25mm/75%/470.hdf5",
        "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,25mm/75%/451.hdf5"
    ]
    get_data_and_plot(files, savename)