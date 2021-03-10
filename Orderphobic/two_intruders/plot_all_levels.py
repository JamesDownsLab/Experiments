import numpy as np
import matplotlib.pyplot as plt

import os

direc = "/media/data/Data/Orderphobic/TwoIntruders/LevelSampling"

files = os.listdir(direc)
files = [f for f in files if f.endswith('.txt') and f.startswith('x')]

bins = np.loadtxt(f"{direc}/bins.txt")

fig, ax = plt.subplots(9, 9, squeeze=True)
for i, leg1 in enumerate(np.arange(-360, 450, 90)):
    for j, leg2 in enumerate(np.arange(-360, 450, 90)):
        x = np.loadtxt(f"{direc}/x_{leg1}_{leg2}.txt")
        freq, _ = np.histogram(x, bins=bins)
        ax[i, j].bar(bins[:-1], freq, width=bins[1]-bins[0])
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])
        if i == 0:
            lbl = f"leg2\n{leg2}" if j == 4 else leg2
            ax[i, j].set_title(lbl)
        if j == 0:
            lbl = f"leg1\n{leg1}" if i == 4 else leg1
            print(lbl)
            ax[i, j].set_ylabel(lbl)
fig.subplots_adjust(wspace=0, hspace=0)
plt.show()