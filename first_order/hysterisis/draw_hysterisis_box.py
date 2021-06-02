from particletracking import dataframes
from labvision import video, images
import numpy as np

import filehandling

file_store = "/media/data/Data/FirstOrder/Hysterisis/FlatPlate/Trial2/0.2_up_1.hdf5"
file_vid = "/media/data/Data/FirstOrder/Hysterisis/FlatPlate/Trial2/0.2_up_1.MP4"

metadata = dataframes.load_metadata(file_store)
vid = video.ReadVideo(file_vid)

box = metadata['crop']

frame = vid.read_frame(vid.num_frames-1)
frame = images.crop(frame, box)

xmin = 750
xmax = 1250
ymin = 750
ymax = 1250

vertices = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])

frame = images.draw_polygon(frame, vertices, thickness=3)
images.save(frame, "/media/data/Data/FirstOrder/Hysterisis/FlatPlate/Trial2/HexagonFigures/box_end.png")
images.display(frame)