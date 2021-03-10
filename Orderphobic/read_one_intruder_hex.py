import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/OneIntruderEverywhere/Logging/100221_600_1_x.txt")
y = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/OneIntruderEverywhere/Logging/100221_600_1_y.txt")

fig, ax = plt.subplots()
hb = ax.hexbin(x, y)
# plt.plot(x, y, alpha=0.5)
# plt.scatter(x, y/, c=np.arange(len(x)))
cb = fig.colorbar(hb, ax=ax)
plt.show()