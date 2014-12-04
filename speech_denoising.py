# Code adapted from Single Channel Noise Suppression for Speech Enhancement available at http://plaza.ufl.edu/hejiaxiu/code.zip by Jiaxiu He
# References:
# [1] Speech Database: http://cslu.cse.ogi.edu/nsel/data/SpEAR_database.html
# [2] Speech Enhancement: Concept and Methodology: http://cslu.cse.ogi.edu/nsel/data/SpEAR_database.html
# [3] Speech Enhancement, a project report: http://web.mit.edu/sray/www/btechthesis.pdf
# [4] Kalman Filtering and Speech Enhancement: http://cmp.felk.cvut.cz/~kybic/dipl/

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import sys


def segment_windows(signal, ww, ov):
    '''
    Parameters
        signal  - array of normalized signal samples
        ww - windwow width
        ov - overlap ratio of windows
    '''
    l = len(signal)
    d = 1 - ov
    frames = int(np.floor((l - ww) / ww / d) + 1)
    seg = np.zeros((ww, frames))

    for i in range(frames):
        start = i * ww * ov
        stop = start + ww
        seg[:, i] = signal[start:stop] * np.hamming(ww)
    return seg, frames

# User Parameters

# FFT
window_length = 256
overlap_ratio = 0.5
# Noise  spectrum
alpha = 0.9
beta = 2
eta = 1.5

# Noise Filtering
sw1 = 4
sw2 = 4
lw1 = 7
lw2 = 7
lmda = 5

csig = wavfile.read('./data/car_clean_lom.wav')  # sampled clean signal
fs, ss = wavfile.read('./data/car_lom.wav')  # sampled obeserved signal
ss = ss / float(np.power(2, 15))  # normalisation

ssw, frames = segment_windows(ss, window_length, overlap_ratio)

sfft = np.fft.fft(ssw, axis=0)
sfftmag = np.abs(sfft)
sfftphase = np.zeros(sfft.shape)
for i in range(frames):
    sfftphase[:, i] = np.angle(sfft[:, i])

# noise spectrum approximation
noise_spectrum = np.zeros(ssw.shape)
noise_spectrum[:, 0] = sfftmag[:, 0]
for i in range(window_length):
    for f in range(1, frames):
        if(sfftmag[i, f] > beta * noise_spectrum[i, f - 1]):
            noise_spectrum[i, f] = noise_spectrum[i, f - 1]
        else:
            noise_spectrum[i, f] = (
                1 - alpha) * sfftmag[i, f] + alpha * noise_spectrum[i, f - 1]

for f in range(frames):
    noise_spectrum[:, f] = np.mean(
        noise_spectrum[:, f:min(f + 11, frames)], axis=1) * eta

# spectral subtraction
specsub = sfftmag - noise_spectrum
negatives = specsub <= 0
specsub[negatives] = np.abs(specsub[negatives] * np.power(10, -3))


# time + freqency filter looking for isolated peaks
for f in range(0,window_length,sw1):
    for i in range(0,frames,sw2):
    	sw = specsub[max(0,f-sw1):min(window_length,f+sw1),max(0,f-sw2):min(window_length,f+sw2)]
    	lw = specsub[max(0,f-lw1):min(window_length,f+lw1),max(0,f-lw2):min(window_length,f+lw2)]

plt.figure()
plt.plot(sfftmag[:, 550])
plt.figure()
plt.subplot(211)

plt.stem(noise_spectrum[:, 550])
plt.subplot(212)
plt.stem(specsub[:, 550])

plt.show()
