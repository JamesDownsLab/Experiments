from labvision import images, video
from particletracking import dataframes
import numpy as np
import os


def check(filename):
    core_name = os.path.splitext(filename)[0]
    vid_name = core_name + '.MP4'
    data_name = core_name + '.hdf5'
    out_name = core_name + '_check_order.png'
    data = dataframes.DataStore(data_name)
    crop = data.metadata['crop']
    vid = video.ReadVideo(vid_name)
    print(vid_name)
    frames = np.arange(4)*vid.num_frames//4
    ims = [images.crop(vid.read_frame(f), crop) for f in frames]
    circles = [data.get_info(f, ['x', 'y', 'r', 'order_long']) for f in frames]
    new_ims = [images.draw_circles_with_scale(im, c[:, :3], c[:, 3]) for im, c in zip(ims, circles)]
    out = images.vstack(images.hstack(new_ims[0], new_ims[1]),
                        images.hstack(new_ims[2], new_ims[3]))
    images.save(out, out_name)

if __name__ == "__main__":
    check("/media/data/Data/FirstOrder/Susceptibility/Flat2/17500010.MP4")