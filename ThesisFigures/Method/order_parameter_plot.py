import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


positions_perfect = np.array([
    [0,0],
    [0.87, 0.5],
    [0, 1],
    [-0.87, 0.5],
    [-0.87, -0.5],
    [0, -1],
    [0.87, -0.5]

])

positions_rough = np.array([
    [0, 0],
    [0.77, 0.66],
    [0, 1],
    [-0.73, 0.65],
    [-0.6, -0.58],
    [0, -1],
    [0.69, -0.65]
])

def set_axis_props(ax):
    ax.set_aspect('equal')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, tight_layout=True)


ax1.plot(positions_perfect[:, 0], positions_perfect[:, 1], 'o')
ax1.plot([0, 2], [0, 0], 'k--')
arc1 = Arc((0, 0), 2, 2, theta1=0, theta2=30)
ax1.add_patch(arc1)
ax1.plot((0, 0.87), (0, 0.5), 'k')
ax1.text(0.5, 0.1, r'$\theta$')
ax1.axis('equal')
ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.5, 1.5])
ax1.set_title('(a)')
set_axis_props(ax1)

ax2.plot(positions_rough[:, 0], positions_rough[:, 1], 'o')
ax2.plot([0, 2], [0, 0], 'k--')
arc2 = Arc((0, 0), 1.96, 1.96, theta1=0, theta2=138.21)
ax2.add_patch(arc2)
ax2.plot((0, -0.73), (0, 0.65), 'k')
ax2.text(0, 0.3, r'$\theta$')
ax2.axis('equal')
ax2.set_xlim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax2.set_title('(b)')
set_axis_props(ax2)

angles_perfect = (np.array([30, 90, 150, 210, 270, 330]))*np.pi/180
angles_rough = (np.array([40.76, 90, 138.21, 224.29, 270, 316.34]))*np.pi/180

orders_perfect = np.cumsum(np.exp(1j*6*angles_perfect))
orders_rough = np.cumsum(np.exp(1j*6*angles_rough))

orders_perfect = np.insert(orders_perfect, 0, 0+0j)
orders_rough = np.insert(orders_rough, 0, 0+0j)

fig.tight_layout()



ax3.plot(np.abs(orders_perfect.real), np.abs(orders_perfect.imag), 'x-')
ax3.plot(np.abs(orders_perfect.real[:2]), np.abs(orders_perfect.imag[:2]), 'r')
ax3.set_xlabel('Re')
ax3.set_ylabel('Im')
ax3.set_ylim([-3, 3])
# set_axis_props(ax3)
ax3.set_title('(c)')

ax4.plot(np.abs(orders_rough.real), np.abs(orders_rough.imag), 'x-')
ax4.plot(np.abs(orders_rough.real)[[2,3]], np.abs(orders_rough.imag)[[2,3]], 'r')
ax4.set_xlabel('Re')
ax4.set_ylabel('Im')
ax4.set_ylim([-1, 1])
# set_axis_props(ax4)
ax4.set_title('(d)')

plt.savefig('OrderParam.jpg')

# plt.show()
`