function err = lpe(x,order)
%% Linear Prediction Error normalized with respect to the energy of the
%% signal. 
%% order = linear prediction order

a = lpc(x,order);
est_x = filter([0 -a(2:end)],1,x);      % Estimated signal
e = (x-est_x)'*(x-est_x);              % Prediction error
ener = x'*x;                   % Energy of the signal
err = e/ener;                           % Linear prediction error