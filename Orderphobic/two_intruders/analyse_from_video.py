from labvision import video, images
from scipy import ndimage

liquid_file = "/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/080121_liquid_580/out.mp4"

solid_file = "/media/data/Data/Orderphobic/TwoIntruders/SpikyIntruder/Logging/060121_solid_600_more_balls/out.mp4"

liquid = video.ReadVideo(liquid_file)
solid = video.ReadVideo(solid_file)

NL = liquid.num_frames
NS = solid.num_frames

N = min(NL, NS)

def get_crop_result(vid):
    frame = vid.read_next_frame()
    frame = ndimage.rotate(frame, -120, reshape=False)
    crop_result = images.crop_rectangle(frame)
    vid.set_frame(0)
    return crop_result

def fix_frame(frame):
    frame = ndimage.rotate(frame, -120, reshape=False)
    frame = images.crop_and_mask(frame, crop_result.bbox, crop_result.mask)
    return frame

crop_result = get_crop_result(liquid)


window = images.Displayer('frames')

for f in range(N):
    liquid_frame = liquid.read_next_frame()
    liquid_frame = fix_frame(liquid_frame)
    solid_frame = solid.read_next_frame()
    solid_frame = fix_frame(solid_frame)

    frames = images.vstack(liquid_frame, solid_frame)
    window.update_im(frames)



