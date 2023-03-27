import pandas as pd
import matplotlib.pyplot as plt


direc = "/media/NAS/share/James/SimulationDataSep22"
for density in [90, 87, 85, 82, 80]:
    for amplitude in ['3e-4', '3,5e-4', '4e-4']:
        filename = f"{direc}/{density}_{amplitude}_savedata"
        data_mean = pd.read_csv(filename+'.mean')
        data_median = pd.read_csv(filename+'.median')
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title(f"{density}_{amplitude}")
        ax.plot()
