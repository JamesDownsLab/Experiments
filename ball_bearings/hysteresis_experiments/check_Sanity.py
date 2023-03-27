import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data1 = pd.read_hdf("/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/2,42mm/0,1_2_down.hdf5")

frame = data1.loc[0]

plt.subplot(1, 2, 1)
plt.scatter(frame.x, frame.y, c=np.abs(frame.hexatic_order))
plt.colorbar()
plt.subplot(1, 2, 2)
plt.scatter(frame.x, frame.y, c=np.angle(frame.hexatic_order), cmap='hsv')
plt.show()