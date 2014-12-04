function y_seg = seg(y, winleng, overate)

%% segment input data y parts with length = winleng and overlapping rate =
%% overate using Hanning window.

dataleng = length(y);
s = 1/(1 - overate);
framenum = floor(s*(dataleng - winleng + 1) / winleng) + 1;    %% number of frames
y_seg = zeros(winleng,framenum);

for k = 0:framenum-1
    a = k*winleng/s+1;
    b = k*winleng/s + winleng;
    segsample = y(a:b).*hamming(winleng);
    y_seg(:,k+1) = segsample;
end