from particletracking import preprocessing, dataframes, tracking
from labvision import images
from numba import jit
import matplotlib.path as mpath
import numpy as np
from moviepy.editor import AudioFileClip

MY_PARAMETERS = {
    'crop method': 'manual',
    'no_of_sides': 6,
    'method': ('flip', 'crop_and_mask', 'grayscale'),
    'min_dist': [27, 3, 51, 1],
    'p_1': [203, 1, 255, 1],
    'p_2': [4, 1, 20, 1],
    'min_rad': [17, 1, 101, 1],
    'max_rad': [17, 1, 101, 1]
}


class HoughManager:
    
    def __init__(self, crop_result=None):
        self.parameters = MY_PARAMETERS
        if crop_result is not None:
            self.parameters['crop method'] = 'batched'
            self.parameters['crop'] = crop_result.bbox
            self.parameters['mask image'] = crop_result.mask
            self.parameters['boundary'] = crop_result.points
        self.preprocessor = preprocessing.PreProcessor(
            self.parameters
        )
    
    def process(self, frame):
        frame, boundary, cropped_frame = self.preprocessor.process(frame)
        return frame, boundary, cropped_frame
    
    def analyse_frame(self, frame):
        new_frame, boundary, cropped_frame = self.process(frame)
        circles = images.find_circles(
            new_frame,
            self.parameters['min_dist'][0],
            self.parameters['p_1'][0],
            self.parameters['p_2'][0],
            self.parameters['min_rad'][0],
            self.parameters['max_rad'][0]
        )
        circles = get_points_inside_boundary(circles, boundary)
        circles = check_circles_bg_color(circles, new_frame, 150)
        return circles, boundary, ['x', 'y', 'r']

    def extra_steps(self, filename):
        vid_name = filename + ".MP4"
        data_name = filename + ".hdf5"
        with dataframes.DataStore(data_name, True) as d:
            f = d.metadata['num_frames']
            duty_cycle = read_audio_file(vid_name, f)
            d.add_frame_property('Duty', duty_cycle)

@jit
def check_circles_bg_color(circles, image, threshold):
    """
    Checks the color of circles in an image and returns white ones

    Parameters
    ----------
    circles: ndarray
        Shape (N, 3) containing (x, y, r) for each circle
    image: ndarray
        Image with the particles in white

    Returns
    -------
    circles[white_particles, :] : ndarray
        original circles array with dark circles removed
    """
    circles = np.int32(circles)
    (x, y, r) = np.split(circles, 3, axis=1)
    r = int(np.mean(r))
    ymin = np.int32(np.squeeze(y-r/2))
    ymax = np.int32(np.squeeze(y+r/2))
    xmin = np.int32(np.squeeze(x-r/2))
    xmax = np.int32(np.squeeze(x+r/2))
    all_circles = np.zeros((r, r, len(xmin)))
    for i, (x0, x1, y0, y1) in enumerate(zip(xmin, xmax, ymin, ymax)):
        im = image[y0:y1, x0:x1]
        all_circles[0:im.shape[0], :im.shape[1], i] = im
    circle_mean_0 = np.mean(all_circles, axis=(0, 1))
    out = circles[circle_mean_0 > threshold, :]
    return out


def get_points_inside_boundary(points, boundary):
    """
    Returns the points from an array of input points inside boundary

    Parameters
    ----------
    points: ndarray
        Shape (N, 2) containing list of N input points
    boundary: ndarray
        Either shape (P, 2) containing P vertices
        or shape 3, containing cx, cy, r for a circular boundary

    Returns
    -------
    points: ndarray
        Shape (M, 2) containing list of M points inside the boundary
    """
    centers = points[:, :2]
    if len(np.shape(boundary)) == 1:
        vertices_from_centre = centers - boundary[0:2]
        points_inside_index = np.linalg.norm(vertices_from_centre, axis=1) < \
            boundary[2]
    else:
        path = mpath.Path(boundary)
        points_inside_index = path.contains_points(centers)
    points = points[points_inside_index, :]
    return points


def read_audio_file(file, frames):
    wav = extract_wav(file)
    wav_l = wav[:, 0]
    # wav = audio.digitise(wav)
    freqs = frame_frequency(wav_l, frames, 48000)
    d = (freqs - 1000)//15
    return d


def digitise(sig):
    """Makes a noisy square signal, perfectly square."""
    out = np.zeros(len(sig))
    out[sig < 0.8*np.min(sig)] = -1
    out[sig > 0.8*np.max(sig)] = 1
    out[(sig > 0.8*np.min(sig))*(sig < 0.8*np.max(sig))] = 0
    return out


def fourier_transform_peak(sig, time_step):
    """Find the peak frequency in a signal"""
    ft = abs(np.fft.fft(sig, n=50000))
    # freq = np.fft.fftfreq(len(sig), time_step)
    freq = np.fft.fftfreq(50000, time_step)
    peak = np.argmax(ft)
    return abs(freq[peak])


def frame_frequency(wave, frames, audio_rate):
    """Returns the peak frequency in an audio file for each video frame"""
    window = int(len(wave)/frames)
    windows = frames
    freq = np.zeros(windows)
    for i in range(windows):
        b = i*window
        t = (i+1)*window
        if t > len(wave):
            t = len(wave)
        freq[i] = int(fourier_transform_peak(wave[b:t], 1/audio_rate))
    return freq


def extract_wav(file):
    audioclip = AudioFileClip(file)
    audioclip_arr = audioclip.to_soundarray(fps=48000, nbytes=2)
    return audioclip_arr


if __name__ == "__main__":
    file = "/media/data/Data/N32/PhaseDiagram_2021_07_06/1700/595.MP4"
    tracker = tracking.ParticleTracker(file, HoughManager(), multiprocess=False)
    tracker.track()