from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import matplotlib.mlab as mlab
import sys
import scikits.audiolab


def segment_windows(signal, ww, ov):
    '''
    Parameters
        signal  - array of normalized signal samples
        ww - window width
        ov - overlap ratio of windows
    '''
    l = len(signal)
    d = 1 - ov
    frames = int(np.floor((l - ww) / ww / d) + 1)
    seg = np.zeros((ww, frames))

    for i in range(frames):
        start = i * ww * ov
        stop = start + ww
        s = signal[start:stop] * np.hamming(ww)
        s[0:16] = 0
        s[ww - 16:ww] = 0
        seg[:, i] = s

    return seg, frames

window_length = 256
overlap_ratio = 0.5  # [0:0.5]
delta = 0.85

cfs, clean = wavfile.read('f16_clean_lom.wav')
fs, ss = wavfile.read('f16_lom.wav')  # sampled observed signal
ss = ss / np.power(2, 15)


ssw, frames = segment_windows(ss, window_length, overlap_ratio)
dataleng = window_length * (1 - overlap_ratio) * (frames - 1) + window_length

plt.figure()
plt.xlabel('t (s)')
plt.ylabel('dB')
plt.yscale('log')
sfft = np.fft.fft(ssw, axis=0)
smag = np.power(np.abs(sfft),2)
plt.plot(smag[89,:],'--')

for i in range(1, frames):
    smag[:, i] = np.sqrt(
        (1 - delta) * np.power(smag[:, i - 1], 2) + delta * np.power(smag[:, i], 2))

plt.plot(smag[89,:])

bin = smag[89,:]
new_bin = np.zeros(bin.shape)
for i in range(frames):

    new_bin[i] = bin[np.argmin(bin[i:min(i+96,frames)])+i]


plt.plot(new_bin,'r',linewidth=2)
plt.show()