import numpy as np

import matplotlib.pyplot as plt

x600 = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/Logging/BallsxLog600_fixed.txt")
x680 = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/Logging/BallsxLog680_fixed.txt")
x700 = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/Logging/BallsxLog700_fixed.txt")



diff_600 = np.abs(x600[1::2] - x600[::2])
diff_680 = np.abs(x680[1::2] - x680[::2])
diff_700 = np.abs(x700[1::2] - x700[::2])

freq_600, bins_600 = np.histogram(diff_600, bins=np.linspace(50, 300, 100), normed=True)
freq_680, bins_680 = np.histogram(diff_680, bins=np.linspace(50, 300, 100), normed=True)
freq_700, bins_700 = np.histogram(diff_700, bins=np.linspace(50, 300, 100), normed=True)

plt.plot(bins_600[:-1], freq_600, label='600')
plt.plot(bins_680[:-1], freq_680, label='680')
plt.plot(bins_700[:-1], freq_700, label='700')
plt.xlabel('Separation (pix)')
plt.ylabel('Relative Frequency')
plt.legend()
plt.show()