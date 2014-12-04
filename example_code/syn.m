function y = syn(y_seg,overate)

%% To synthesis signals from signal segments

[winleng,framenum] = size(y_seg);
dataleng = winleng*(1-overate)*(framenum - 1) + winleng;
y = zeros(dataleng,1);
for k = 0:framenum-1
    a = k*winleng*(1-overate)+1;
    b = a+winleng-1;
    y(a:b) = y(a:b) + y_seg(:,k+1);
end