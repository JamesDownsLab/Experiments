import numpy as np
import matplotlib.pyplot as plt

XMIN = 100
XMAX = 700

file = "/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/210121_liquid_{}_flipped_rail.txt"
duties = [590, 600, 610, 620]

fig, ax = plt.subplots(2, 2, sharex=True)
ax = np.ndarray.flatten(ax)

bins = np.arange(XMIN, XMAX)
for i, d in enumerate(duties):
    data = np.loadtxt(file.format(d))
    ax[i].hist(data, bins=bins)
    ax[i].set_title(d)

plt.show()
