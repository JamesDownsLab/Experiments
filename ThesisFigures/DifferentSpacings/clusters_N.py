import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 18})

r = 0.58
dr = 0.03 # error on r

Ls = np.arange(1.85, 1.971, 0.001)
dL = 0.01 # error on L
D = 2.00

Nmaxs = []
Nmins = []
Ns = []

for L in Ls:
    N = (D - L + r) / (D - L)  # correct N
    Ns.append(N)

    ri = 0.55, 0.61  # min and max r
    Li = L - dL, L + dL  # min and max L

    Nrange = []
    for rj in ri:
        for Lj in Li:
            N = (D - Lj + rj) / (D - Lj)
            Nrange.append(N)

    Nmaxs.append(max(Nrange))
    Nmins.append(min(Nrange))

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot(Ls, Ns)
ax.fill_between(Ls, Nmins, Nmaxs, alpha=0.4)
ax.set_xlabel('L [mm]')
ax.set_ylabel('$N_{exp}$')
ax.set_xticks([1.86, 1.88, 1.90, 1.92, 1.94, 1.96])
ax.set_xlim([1.85, 1.97])
fig.tight_layout()

plt.savefig("/media/data/Data/BallBearing/HIPS/IslandExperiments/N_expected.png", dpi=600)

