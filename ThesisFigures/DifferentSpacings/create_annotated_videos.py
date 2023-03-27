import pandas as pd
import numpy as np
import filehandling
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

directory = "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,10mm/65%"

directories = [
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,10mm/65%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,10mm/75%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,10mm/85%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,00mm/65%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,00mm/75%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,00mm/85%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,25mm/65%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,25mm/75%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,25mm/85%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,31mm/65%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,31mm/75%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,31mm/85%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,42mm/65%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,42mm/75%",
    "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,42mm/85%",
]



def run(direc):
    def update_plot(d):
        df = d[0]
        duty = d[1]
        for artist in ax.collections:
            artist.remove()
        angles = np.angle(df.hexatic_order.values)
        col = cmap(norm(angles))
        ax.quiver(df.x, df.y, np.real(df.hexatic_order),
                  np.imag(df.hexatic_order),
                  color=col, pivot='mid')
        ax.set_title(duty)

    def create_animation_window():
        fig, (ax, cax) = plt.subplots(1, 2,
                                      gridspec_kw={'width_ratios': [50, 1]})
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        cmap = matplotlib.cm.hsv
        norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                                orientation='vertical')
        return fig, ax, cmap, norm

    print(direc)
    files = filehandling.get_directory_filenames(f"{directory}/*.hdf5")
    fig, ax = plt.subplots()
    data = [[pd.read_hdf(f).loc[0], int(f[-8:-5])] for f in files[::-1]]
    fig, ax, cmap, norm = create_animation_window()
    ani = FuncAnimation(fig, update_plot, frames=data)

    ani.save(f"{directory}/animation.mp4", dpi=600)

for directory in directories:
    run(directory)