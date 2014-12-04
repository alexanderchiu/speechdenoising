% Use the adaptive noise spectral estimation, not VAD.
%% Use the time-frequency filtering to reduce musical noise.

clear;
clc;

%% Segmentation
cleansp = wavread('car_clean_lom.wav');        %% clean speech
[rawsig,fs] = wavread('car_lom.wav');  %% raw signals
winleng = 256;                          %% Window length
overate = 0.5;                          %% Overlapping rate
rawsig_seg = seg(rawsig,winleng,overate);    %% segmented raw signals
framenum = size(rawsig_seg,2);          %% # of frames
dataleng = winleng*(1-overate)*(framenum - 1) + winleng;
%% data length after segmentation


%% Noisy speech spectral estimation
sigfft = fft(rawsig_seg);       %% fft of segmented signal
sigphase=zeros(size(sigfft));   %% phase of the noisy speech 
for k=1:framenum;
    sigphase(:,k) = angle(sigfft(:,k));
end
sigmag = abs(sigfft);           %% spectral magnitude

%% Adaptive noise spectral estimation
noimag = zeros(winleng,framenum);   %% original noise magnitude estimates 
noimag(:,1) = sigmag(:,1);          
    %% suppose the first frame only contains noise. 
alpha = 0.9;
    %% N_(k) = (1-alpha)*X_i(k) + alpha*N_i(k-1)
beta = 2;
    %% test if X_i( k) > beta*N_i(k-1)

for l = 1:winleng   %% for each freq bin ( fft length = window length)
    for k = 2:framenum  %% for each frame
        sigmag(l,k);
        beta*noimag(l,k-1);
        if sigmag(l,k) > beta*noimag(l,k-1)
            noimag(l,k) = noimag(l,k-1);
        else
            noiest = (1-alpha)*sigmag(l,k) + alpha*noimag(l,k-1);
            noimag(l,k) = noiest;
        end
    end
end

for k = 1:framenum
    noimag(:,k) = mean(noimag(:,k:min(k+10,framenum)),2);
end
noimag = noimag*1.5;    
    %% modified noise spectrum estimates
    %% = original estimates * overestimation factor 

magaver = sigmag;


%% Noise subtraction and half-wave rectification
magaver = sigmag;
magtil = magaver - noimag;
% magtil = max(magtil, zeros(winleng, framenum));
for k = 1:framenum
    for l = 1:winleng
        if magtil(l,k)<0
            magtil(l,k) = abs(magtil(l,k))*10^(-3);
        end
    end
end 

%% Time-frequency filtering
a1 = 7;
a2 = 7;
b1 = 4;
b2 = 4;
lambda = 5;
m = 0;
for l = 1:b1:winleng
    for k = 1:b2:framenum
        max(1,l-b1), min(winleng,l+b1), max(1,k-b2),min(framenum,k+b2)
        regb = magtil( max(1,l-b1):min(winleng,l+b1), max(1,k-b2):min(framenum,k+b2));
        rega = magtil( max(1,l-a1):min(winleng,l+a1), max(1,k-a2):min(framenum,k+a2));
        pb = sum(sum(regb));
        pa = sum(sum(rega)) - pb;
        if pb >= lambda*pa
            m=m+1;
            index(m,1) = l;
            index(m,2) = k;
        end
    end
end

for n = 1:m
    freqbin = index(n,1);
    frameind = index(n,2);
    t1 = max(frameind-b2,1);
    t2 = min(frameind+b2,framenum);
    f1 = max(freqbin-b1,1);
    f2 = min(freqbin+b1,winleng);
    magtil(f1:f2,t1:t2) = zeros(length(f1:f2),length(t1:t2))*10^(-4);
end

% %% Smoothing
% delta = 0.9;
% for k=2:framenum
%     magtil(:,k) = ( (1-delta)*magtil(:,k-1).^2+delta*magtil(:,k).^2 ).^(.5);
% end

% %% Synthesis
% sighat = magtil.*exp(i*sigphase);
% sigest_seg = real( ifft(sighat) );
% % noiest_seg = real( ifft(noimag) );
% % for p =1:length(nindex)
% %     k = nindex(p);
% %     sigest_seg(:,k) = sigest_seg(:,k)*10^(-1);
% % end

% % lpccoef = lpc(sigest_seg,13);
% % for k = 1:framenum
% %     sigest2_seg(:,k) = filter([0,-lpccoef(k,2:end)],1,sigest_seg(:,k));
% % end
% % 
% sigest = real(syn(sigest_seg,overate));
% % noiest = real(syn(noiest_seg,overate));
% snr_out = 10*log(sum(sigest.^2) / sum((rawsig(1:dataleng) - sigest).^2))



% % Plot
% t = (1:dataleng)/fs;
% figure(1);
% subplot(3,1,1), plot(t,cleansp(1:dataleng));
% xlabel('Time (s)');
% title('Clean speech');
% subplot(3,1,2), plot(t,rawsig(1:dataleng));
% xlabel('Time (s)');
% title('Noisy speech');
% subplot(3,1,3),
% plot(t,sigest);
% xlabel('Time (s)');
% title('Enhanced speech');

% figure(2)
% subplot(3,1,1),specgram(cleansp(1:dataleng),256,fs);
% title('Clean speech');
% subplot(3,1,2),specgram(rawsig(1:dataleng),256,fs);
% title('Noisy speech');
% subplot(3,1,3),
% specgram(sigest,256,fs);
% title('Enhanced speech');
% % soundsc(cleansp,fs);
% % soundsc(rawsig,fs);
% % pause;
% % soundsc(sigest,fs);
% wavwrite(sigest,fs,'car_enhanced_m2.wav');
