import os
import warnings

import filehandling
from particletracking import tracking
import track
from tqdm import tqdm
from labvision import images, video
from particletracking import dataframes, statistics



def get_crop_result(file):
    vid = video.ReadVideo(file)
    frame = vid.read_next_frame()
    return images.crop_polygon(frame)

def go(files, crop_result):
    for i, file in tqdm(enumerate(files)):
        # file = direc + '/' + file
        name, ext = os.path.splitext(file)
        if ext == '.MP4':
            if i == 0 and crop_result is None:
                print('cropping')
                crop_result = get_crop_result(file)
            data_file = name + '.hdf5'
            if not os.path.exists(data_file):
                tracker = tracking.ParticleTracker(file, track.HoughManager(crop_result=crop_result), True)
                tracker.track()
                # tracker.link()
    return crop_result

parent = "/media/data/Data/N32/PhaseDiagram_2021_07_06"
all_files = []
for direc in os.listdir(parent):
    new_direc = parent + '/' + direc
    print(new_direc)
    files = filehandling.get_directory_filenames(new_direc+'/*.MP4')
    all_files += files

print(all_files)
go(all_files, None)