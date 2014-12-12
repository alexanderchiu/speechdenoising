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


def combine_segemnts(segments, ov):
    '''
    Parameters
        signal  - array of normalized signal samples
        ov - overlap ratio of windows
    '''
    ww, frames = segments.shape
    dataleng = ww * (1 - ov) * (frames - 1) + ww

    sig = np.zeros(dataleng)
    for i in range(frames):
        start = i * ww * (1 - ov)
        stop = start + ww
        sig[start:stop] = sig[start:stop] + segments[:, i]
    return sig

# User Parameters

# Interface
show_graphs = True
play_estimate = True

# FFT
window_length = 256
overlap_ratio = 0.5  # [0:0.5]

# Noise  spectrum
alpha = 0.75  # [0.75:0.95]
beta = 2  # [1.5:2.5]
eta = 1.5  # [1.0:2.0]

# Noise Filtering
sw1 = 4
sw2 = 4
lw1 = 7
lw2 = 7
lmda = 5

# Smoothing
delta = 0.85

cfs, clean = wavfile.read('./data/car_clean_lom.wav')
fs, ss = wavfile.read('./data/hynek.wav')  # sampled observed signal
ss = ss / np.power(2, 15)

print fs
ssw, frames = segment_windows(ss, window_length, overlap_ratio)
dataleng = window_length * (1 - overlap_ratio) * (frames - 1) + window_length

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
specsub[negatives] = np.abs(specsub[negatives]) * pow(10, -3)

peaks = []
# time + frequency filter looking for isolated peaks
for f in range(0, window_length, sw1):
    for i in range(0, frames, sw2):
        sw = specsub[max(0, f - sw1):min(window_length, f +
                                         sw1 + 1), max(0, i - sw2):min(frames, i + sw2 + 1)]
        lw = specsub[max(0, f - lw1):min(window_length, f +
                                         lw1 + 1), max(0, i - lw2):min(frames, i + lw2 + 1)]
        psw = np.sum(sw)
        plw = np.sum(lw) - psw
        if(psw > lmda * plw):
            peaks.append((f, i))

# Zero isolated peaks
for peak in peaks:
    f = peak[0]
    t = peak[1]
    t1 = max(0, t - sw2)
    t2 = min(frames, t + sw2 + 1)
    f1 = max(0, f - sw1)
    f2 = min(window_length, f + sw1 + 1)
    specsub[f1:f2, t1:t2] = np.zeros((f2 - f1, t2 - t1)) * pow(10, -4)

# Smoothing
for i in range(1, frames):
    specsub[:, i] = np.sqrt(
        (1 - delta) * np.power(specsub[:, i - 1], 2) + delta * np.power(specsub[:, i], 2))

# calculate enhanced speech from spectrum
estim_spec = specsub * np.exp(1j * sfftphase)
estim_seg = np.real(np.fft.ifft(estim_spec, axis=0))
estim = combine_segemnts(estim_seg, overlap_ratio)

# Pretty figures
if(show_graphs):
    t = np.linspace(0, dataleng, dataleng) / fs
    plt.subplot(3, 3, 1),

    plt.plot(t, clean[:dataleng])
    plt.xlabel('Time (s)')
    plt.title('Clean speech')
    plt.xlim(0, max(t))
    plt.subplot(3, 3, 4)
    plt.plot(t, ss[:dataleng])
    plt.xlabel('Time (s)')
    plt.title('Observed speech')
    plt.xlim(0, max(t))
    plt.subplot(3, 3, 7),
    plt.plot(t, estim)
    plt.xlabel('Time (s)')
    plt.title('Denoised speech')
    plt.xlim(0, max(t))

    cfft = np.abs(np.fft.fft(clean))
    ssfft = np.abs(np.fft.fft(ss))
    estimfft = np.abs(np.fft.fft(estim))
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(3, 3, 2)
    plt.plot(np.arange(len(cfft)) / len(cfft), cfft)
    plt.xlabel('$f_w$')
    plt.title('Clean speech')
    plt.subplot(3, 3, 3)
    Pxx, freqs, bins, im = plt.specgram(clean, Fs=fs)
    plt.xlim(0, max(bins))
    plt.ylabel("Frequency (Hz)")
    plt.xlabel('t (s)')
    plt.title('Clean speech')
    plt.subplot(3, 3, 5)
    plt.xlabel('$f_w$')
    plt.title('Observed speech')
    plt.plot(np.arange(len(ssfft)) / len(ssfft), ssfft)

    plt.subplot(3, 3, 6)
    Pxx, freqs, bins, im = plt.specgram(ss, Fs=fs)
    plt.xlim(0, max(bins))
    plt.ylabel("Frequency (Hz)")
    plt.xlabel('t (s)')
    plt.title('Observed speech')
    plt.subplot(3, 3, 8)
    plt.xlabel('$f_w$')
    plt.title('Denoised speech')
    plt.plot(np.arange(len(estimfft)) / len(estimfft), estimfft)
    plt.subplot(3, 3, 9)
    Pxx, freqs, bins, im = plt.specgram(estim, Fs=fs)
    plt.xlim(0, max(bins))
    plt.ylabel("Frequency (Hz)")
    plt.xlabel('t (s)')
    plt.title('Denoised speech')

#play .wav
if(play_estimate):
    scikits.audiolab.play(estim, fs)

#save .wav
wavfile.write('estimate.wav', fs, estim)

plt.show()
