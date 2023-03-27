from labvision import images, video
from particletracking import dataframes
import numpy as np
import os

def check(filename):
    core_name = os.path.splitext(filename)[0]
    vid_name = core_name + '.MP4'
    data_name = core_name + '.hdf5'
    out_name = core_name + '_check.png'
    data = dataframes.DataStore(data_name)
    x = data.get_info(0, ['x'])
    print(len(x))
    crop = data.metadata['crop']
    vid = video.ReadVideo(vid_name)
    print(vid_name)
    frames = np.arange(4)*vid.num_frames//4
    ims = [images.crop(vid.read_frame(f), crop) for f in frames]
    circles = [data.get_info(f, ['x', 'y', 'r']) for f in frames]
    new_ims = [images.draw_circles(im, c) for im, c in zip(ims, circles)]
    out = images.vstack(images.hstack(new_ims[0], new_ims[1]),
                        images.hstack(new_ims[2], new_ims[3]))
    images.save(out, out_name)



if __name__ == "__main__":
    check("/media/data/Data/N29/FirstOrder/Interfaces/RepeatsForFluctuationsNovember2019/16320002.hdf5")