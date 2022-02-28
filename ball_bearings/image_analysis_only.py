from labvision import images, video

vid = video.ReadVideo("/media/data/Data/BallBearing/HIPS/PhaseDiagrams/2,31mm/density70.MP4")

frame = vid.read_next_frame()

crop_result = images.crop_polygon(frame)

cropped = images.crop_and_mask(frame, crop_result.bbox, crop_result.mask)

images.displa