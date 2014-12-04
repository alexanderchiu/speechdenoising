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

def combine_segemnts(segments, ov):
    '''
    Parameters
        signal  - array of normalized signal samples
        ov - overlap ratio of windows
    '''
    ww, frames = segments.shape
    dataleng = ww*(1-ov)*(frames- 1) + ww;

    sig = np.zeros(dataleng);
    for i in range(frames):
        start = i*ww*(1-ov);
        stop = start+ww;
        sig[start:stop] = sig[start:stop] + segments[:,i]
    return sig

# User Parameters

# FFT
window_length = 256
overlap_ratio = 0.5
# Noise  spectrum
alpha = 0.9
beta = 2
eta = 1.5

# Noise Filtering
sw1 = 5
sw2 = 5
lw1 = 10
lw2 = 10
lmda = 5

# Smooting
delta = 0.9;


fs, ss = wavfile.read('./data/DEKF_cellular_0db__noisy.wav')  # sampled obeserved signal
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
specsub[negatives] = np.abs(specsub[negatives])*pow(10,-3)

peaks = []
# time + freqency filter looking for isolated peaks
for f in range(0, window_length, sw1):
    for i in range(0, frames, sw2):
        sw = specsub[max(0, f - sw1):min(window_length, f +
                                         sw1+1), max(0, i - sw2):min(frames, i + sw2+1)]
        lw = specsub[max(0, f - lw1):min(window_length, f +
                                         lw1+1), max(0, i - lw2):min(frames, i + lw2+1)]
        psw = np.sum(sw)
        plw = np.sum(lw) - psw
        if(psw > lmda * plw):
            peaks.append((f, i))

#Zero isolated peaks
for peak in peaks:
    f = peak[0]
    t = peak[1]
    t1 = max(0, t - sw2)
    t2 = min(frames, t + sw2+1)
    f1 = max(0, f - sw1)
    f2 = min(window_length,f + sw1+1)
    specsub[f1:f2, t1:t2] = np.zeros((f2-f1,t2-t1))


for i in range(1,frames):
    specsub[:,i] = np.sqrt((1- delta)*np.power(specsub[:,i-1],2) + delta*np.power(specsub[:,i],2))

estim_spec = specsub*np.exp(1j*sfftphase)

estim_seg = np.real(np.fft.ifft(estim_spec,axis =0))

estim = combine_segemnts(estim_seg,overlap_ratio)

wavfile.write('estimate.wav',fs,estim);
