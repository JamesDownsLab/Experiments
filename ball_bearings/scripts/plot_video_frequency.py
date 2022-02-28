from labvision import audio, video
import matplotlib.pyplot as plt
import numpy as np

import filehandling
file = "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,48mm/19080080.MP4"

wav = audio.extract_wav(file)
# plt.plot(wav[:, 0])
# plt.show()

freq = audio.frame_frequency(wav[:, 0], len(wav)*25//48000, 48000)


plt.plot(freq)
plt.show()

# files = filehandling.list_files("/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,10mm/*")
# freqs = []
# duties = []
# duty = 550
# for file in files:
#     wav = audio.extract_wav(file)
#     freq = audio.frame_frequency(wav[:, 0], len(wav)*25//48000, 48000)
#     freqs.append(np.mean(freq))
#     duties.append(duty)
#     duty -= 1
#
# plt.plot(duties, freqs)
# plt.show()
