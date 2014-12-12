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
        # s[0:16] = 0
        # s[ww - 16:ww] = 0
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
window_length = 256
overlap_ratio = 0.5  # [0:0.5]


cfs, clean = wavfile.read('./data/car_clean_lom.wav')
fs, ss = wavfile.read('./data/hynek.wav')  # sampled observed signal
ss = ss / np.power(2, 15)


ssw, frames = segment_windows(ss, window_length, overlap_ratio)
dataleng = window_length * (1 - overlap_ratio) * (frames - 1) + window_length

plt.figure()
plt.xlabel('$\lambda$')
plt.ylabel('dB')
plt.yscale('log')
plt.xlim(0, frames)

bin = 20
U = 8
V = 12
D = U*V
# Interface
show_graphs = True
play_estimate = True

sfft = np.fft.fft(ssw, axis=0)
sfftmag = np.abs(sfft)
smag = np.power(np.abs(sfft), 2)
sfftphase = np.zeros(sfft.shape)
for i in range(frames):
    sfftphase[:, i] = np.angle(sfft[:, i])
plt.plot(smag[bin, ], '--')

npsd = np.zeros(sfft.shape) # noise power spectral density
spd = np.zeros(sfft.shape) # smoothed power density
fmpsd = np.zeros(sfft.shape) # first moment power spectral density
smpsd = np.zeros(sfft.shape) # second moment power spectral density
var = np.zeros(sfft.shape) # second moment power spectral density
qeq = np.zeros(sfft.shape)
bmin = np.zeros(sfft.shape)
bmin_sub = np.zeros(sfft.shape)

gamma_h = np.zeros(sfft.shape)
alpha = np.zeros(sfft.shape)
beta = np.zeros(sfft.shape)
alpha_hat = np.zeros(frames)
alpha_c = np.zeros(frames)
q = np.zeros(frames)


bc = np.zeros(frames)
k_mod = np.zeros(frames)
lmin_flag = np.zeros(sfft.shape)
actmin = np.zeros(sfft.shape) +999999
actmin_sub = np.zeros(sfft.shape) +999999
store = np.zeros(frames*window_length)
pmin_u = np.zeros(sfft.shape)
noiseslopemax = 0

spd[:, 0] = smag[:, 0]
npsd[:, 0] = spd[:, 0]
# print npsd[:,0]
subwc = V

pmin = np.zeros(sfft.shape)

for i in range(1, frames):
    for k in range(window_length):

        
        # gamma_h[k,i]  = smag[k,i-1]/npsd[k,i]
        alpha_hat[i] = 1/(1+np.power(np.sum(spd[:,i-1])/np.sum(smag[:,i])-1,2))
        alpha_c[i] = 0.7*alpha_c[i-1]+0.3*max(alpha_hat[i] ,0.7)
        alpha[k,i] = max(0.96*alpha_c[i]/(1+np.power(spd[k,i-1]/npsd[k,i-1] -1,2)),0.3)


        spd[k, i] = alpha[k,i]  * spd[k, i - 1] + (1 - alpha[k,i] ) * smag[k, i]

        beta[k,i] = min(np.power(alpha[k,i],2),0.8)
        fmpsd[k, i] = beta[k,i]  * fmpsd[k, i - 1] + (1 - beta[k,i] ) * spd[k, i]
        smpsd [k, i] = beta[k,i]  * smpsd [k, i - 1] + (1 - beta[k,i] ) * np.power(spd[k, i],2)
        
        var[k,i] = smpsd [k, i] - np.power(fmpsd[k, i],2)


        qeq[k,i] = min(var[k,i]/(2*np.power(npsd[k,i-1],2)),0.5)


        q[i] = 1/window_length*np.sum(qeq[:,i],axis = 0)


        bc[i] = 1 + 2.12*np.sqrt(q[i])
        
        bmin[k,i] = (1 + (D-1)*2*q[i])
        # bmin_sub[k,i] = (1 + (V-1)*2*q[i])
        pmin[k, i] = np.min(spd[k, max(0, i - D):i])
        npsd[k,i] =pmin[k,i]*bmin[k,i]*bc[i]
        # k_mod = np.zeros(window_length)

        # if(spd[k,i]*bmin[k,i]*bc[i] < actmin[k,i]):
        #     actmin[k,i] = spd[k,i]*bmin[k,i]*bc[i]
        #     actmin_sub[k,i] = spd[k,i]*bmin_sub[k,i]*bc[i]
        #     k_mod[k] =1

        # if(subwc == V):
        #     if(k_mod[k] == 1):
        #         lmin_flag[k,i] = 0
        #     store[i+k*window_length -1]= actmin[k,i]

        #     pmin_u[k,i] = np.min(store[max(0,i+k*window_length-U):i+k*window_length])
        #     if(q,[i]< 0.03):
        #         noiseslopemax = 8
        #     elif(q[i] <0.05):
        #         noiseslopemax = 4
        #     elif(q[i] <0.06):
        #         noiseslopemax = 2
        #     else:
        #         noiseslopemax = 1.2


        #     if(lmin_flag[k,i] and actmin_sub[k,i] < noiseslopemax*pmin_u[k,i] and actmin_sub[k,i > pmin_u[k,i]]):
        #         pmin_u[k,i] = actmin_sub[k,i]

        #     lmin_flag[k,i] = 0
        #     subwc = 1
        #     actmin_sub[k,i] = 999999
        #     actmin[k,i] = 999999
        # else:
        #     if (subwc > 1):
        #         if(k_mod[k] == 1 ):
        #            lmin_flag[k,i] =1
        #         npsd[k, i] = min(actmin_sub[k,i],pmin_u[k,i])
        #         pmin_u[k,i] = npsd[k,i]
        # subwc = subwc+1


# for i in range(1,frames):
#     npsd[:, i] =0.85*npsd[:, i - 1] + (1 - 0.85) * np.power(npsd[:,i],2)



plt.plot(spd[bin, :], 'g',)
plt.plot(npsd[bin, :], 'r', linewidth=2)



specsub = sfftmag - npsd
negatives = specsub <= 0
specsub[negatives] = np.abs(specsub[negatives]) * pow(10, -3)

# calculate enhanced speech from spectrum
estim_spec = specsub * np.exp(1j * sfftphase)
estim_seg = np.real(np.fft.ifft(estim_spec, axis=0))
estim = combine_segemnts(estim_seg, overlap_ratio)
plt.figure()
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