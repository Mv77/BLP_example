function [vcov] = avar_est(par_opt,S,X,P,LnYSim,Z,W)
% Asymptotic variance estimation following
% Econometrics, Hansen 2019, chapter 13.25.

% This uses a numeric deivative of the moment
% function, computed using the ADD-ON
% "DERIVEST".

% Number of observations
n = length(P);

% Compute the moments at every observation
ret_sum = false;
split_pars = false;
g = moment(par_opt,S,X,P,LnYSim,Z,W,ret_sum, split_pars);

% Average moments
g_bar = mean(g,1);

% Find estimated vcov matrix of the moments.
omega_hat = zeros(size(Z,2));
for i=1:size(g,1)
    
    dev = (g(i,:) - g_bar)';
    omega_hat = omega_hat + dev*dev';

end
omega_hat = omega_hat/n;

% Find jacobian matrix (true,false since we need the sum and are providing
% the full param vector)
Q = jacobianest(@(p) moment(p,S,X,P,LnYSim,Z,W,true,false)/n,par_opt);

% Compute vcov matrix
vcov = (Q'*W*Q)\(Q'*W*omega_hat*W*Q)/(Q'*W*Q);