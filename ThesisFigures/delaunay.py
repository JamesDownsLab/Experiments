import numpy as np
from particletracking.dataframes import DataStore
from labvision import images, video

data = DataStore("/media/data/Data/N29/FirstOrder/Hysterisis/5Repeats/RedTrayDense/0.1_up_1.hdf5")

vid = video.ReadVideo("/media/data/Data/N29/FirstOrder/Hysterisis/5Repeats/RedTrayDense/0.1_up_1.MP4")

#%%
frame = vid.read_next_frame()

xyr = data.get_info(0, ['x', 'y', 'r'])


#%%
crop = data.metadata['crop']
boundary = data.metadata['boundary']

frame = images.crop(frame, crop)

#%%
images.display(frame)

#%%
annotated = images.draw_circles(frame, xyr)
images.display(annotated)

#%%
import matplotlib.pyplot as plt
from scipy import spatial

delaunay = spatial.Delaunay(xyr[:, :2])

spatial.delaunay_plot_2d(delaunay)
plt.imshow(frame)