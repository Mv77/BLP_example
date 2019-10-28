function [loss,est] = GMMloss(par,S,X,P,LnYSim,Z, W, split_par)
% Computes the objective function of the GMM routine at
% parameters "par".

% Number of observations
n = length(P);

% Get the sum of moments over all observations
ret_sum = true;
[g, est] = moment(par,S,X,P,LnYSim,Z, W,ret_sum, split_par);

% Compute the objective function
loss = n*(g/n)'*W*(g/n);
