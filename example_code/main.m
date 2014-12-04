

clear;
clc;

%% Segmentation
cleansp = wavread('car_clean_lom.wav');    %% clean speech
[rawsig,fs] = wavread('car_lom.wav');  %% raw signals
winleng = 256;  %% Window length
overate = 0.5;  %% Overlapping rate
rawsig_seg = seg(rawsig,winleng,overate);    %% segmented raw signals
framenum = size(rawsig_seg,2);
dataleng = winleng*(1-overate)*(framenum - 1) + winleng;

%% FFT
sigfft = fft(rawsig_seg);   %% fft of segmented signal
sigphase=zeros(size(sigfft));   %% phase 
for k=1:framenum;
    sigphase(:,k) = angle(sigfft(:,k));
end
sigmag = abs(sigfft);   %% spectral magnitude
% power_seg = sigfft_seg.*conj(sigfft_seg);   %% power spectrum

%% VAD
D = zeros(1,framenum);
order = 13;     %% LP order
for k=1:framenum
    x = rawsig_seg(:,k);
    ener = x'*x;
    D(k) = ener*( 1-zcr(x) )*( 1-lpe(x,order) );
end
D=D/max(D);
dthresh = 0.05;
nindex = find(D <= dthresh);
sindex = find(D > dthresh);

% % Magnitude Averaging
% magaver = zeros(size(sigfft));
% for k=1:framenum
%     if k<3
%         magaver(:,k) = mean(sigmag(:,1:k),2);
%     else
%         magaver(:,k) = mean(sigmag(:,k-2:k),2);
%     end
% end
magaver = sigmag;
%% Noise Estimation
munoi = mean(magaver(:,nindex),2);
magnoi = munoi*ones(1,framenum);

%% Noise subtraction and half-wave rectification
magtil = magaver - magnoi;
for k=1:framenum
    for l=1:winleng
        if magtil(l,k)<0
            magtil(l,k)=0;
        end
    end
end


% %% Residual noise reduction
% mumax = max(magtil(:,nindex),[],2);
% for l=1:winleng
%     for k=1:framenum
%         if magtil(l,k)<mumax(l)
%             if k==1
%                 magtil(l,k) = min(magtil(l,1),magtil(l,2));
%             elseif k==framenum
%                 magtil(l,k) = min(magtil(l,k-1:k));
%             else
%                 magtil(l,k) = min(magtil(l,k-1:k+1));
%             end
%         end
%     end
% end

% %% Additional niose suppression
% for k=1:framenum
%     T = sum(magtil(:,k)./munoi)/winleng;
%     if T < 10^(-0.6);
%         magtil(:,k) = magtil(:,k)*10^(-1.5);
%     end
% end

%% Smoothing
delta = 0.9;
for k=2:framenum
    magtil(:,k) = (1-delta)*magtil(:,k-1)+delta*magtil(:,k);
end


%% Synthesis
sighat = magtil.*exp(i*sigphase);
sigest_seg = ifft(sighat);
sigest = syn(sigest_seg,overate);
t = (1:dataleng)/fs;
figure(1);
subplot(3,1,1), plot(t,rawsig(1:dataleng));
xlabel('Time (s)');
title('Noisy speech');
subplot(3,1,2), plot(t,cleansp(1:dataleng));
xlabel('Time (s)');
title('Clean speech');
subplot(3,1,3), plot(t,real(sigest));
xlabel('Time (s)');
title('Estimated signal');

figure(2)
subplot(3,1,1),specgram(cleansp(1:dataleng),256,fs);
title('Clean speech');
subplot(3,1,2),specgram(rawsig(1:dataleng),256,fs);
title('Noisy speech');
subplot(3,1,3),specgram(sigest,256,fs);


% soundsc(rawsig,fs);
% pause;
% soundsc(real(sigest),fs);
wavwrite(sigest,fs,'car_enhanced_m1.wav');

% %% Synthesis using phase randomization
% sighat = magtil.*exp(i*sigphase);
% for k = 1:length(nindex)
%     nind = nindex(k);
%     ranphase = sigphase(:,nind) + pi*2*(rand(winleng,1)-0.5);
%     sighat(:,nind) = magtil(:,nind).*exp(i*ranphase);
% end
% sigest_seg = ifft(sighat);
% sigest = syn(sigest_seg,overate);
% plot(rawsig, 'r')
% hold on;
% plot(real(sigest));
% % soundsc(rawsig,fs);
% % pause;
% soundsc(real(sigest),fs);
% wavwrite(sigest,fs,'volvor4_enhanced.wav');