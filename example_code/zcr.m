function [n] = zc(x)
% ZC number of zero crossings in x
%    [n] = zc(x) calculates the number of zero crossings in x
s=sign(x);
t=filter([1 1],1,s);
n=(length(s)-length(find(t)))/length(s);

