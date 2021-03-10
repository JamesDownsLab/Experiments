import os
import warnings

import filehandling
from particletracking import tracking
import track
from tqdm import tqdm
from labvision import images, video
from particletracking import dataframes, statistics

direc1 = filehandling.open_directory('Open Directory containing videos')
files = filehandling.get_directory_filenames(direc1+'/*.MP4')
# direc2 = filehandling.open_directory()
# files2 = filehandling.get_directory_filenames(direc2+'/*.MP4')
# files = files1 + files2
print(files)

def get_crop_result(file):
    vid = video.ReadVideo(file)
    frame = vid.read_next_frame()
    return images.crop_polygon(frame)

for i, file in tqdm(enumerate(files)):
    # file = direc + '/' + file
    name, ext = os.path.splitext(file)
    if ext == '.MP4':
        if i == 0: crop_result = get_crop_result(file)
        data_file = name + '.hdf5'
        if not os.path.exists(data_file):
            tracker = tracking.ParticleTracker(file, track.HoughManager(crop_result=crop_result), True)
            tracker.track()

# files = filehandling.get_directory_filenames(direc+'/*.hdf5')
for file in files:
    data = dataframes.DataStore(file)
    calculator = statistics.PropertyCalculator(data)
    calculator.order()
    # calculator.density()