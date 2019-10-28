function [g, est] = moment(par,S,X,P,LnYSim,Z,W,ret_sum, split_par)
% Computes the moments of interest at given
% parameters "par". Also return the estimates
% for convenience if requested.

% - ret_sum: boolean defining wheter to return
%            the sum of moments across observations
%            or a matrix with each moment at each observation
%
% - split_par: boolean defining whether the linear / non-linar
%              parameter improvement is being used.

if split_par
    alpha1 = par;
else
    [alpha0, alpha1, beta] = vec_2_par(par);
end

% Initialize delta for the contraction mapping.
s0 = 1 - sum(S);
delta0 = log(S) - log(s0);

% Perform demand inversion through contraction mapping
tol = 1e-12;
delta = contr_map(alpha1, delta0, S, P, LnYSim, tol);

if split_par
    % If splitting linear and non-linear parameters, fit the model
    % delta = X*beta - alpha0*P + xi to find optimal beta and alpha.
    design = [X,-P];
    coef = (design'*Z*W*Z'*design)\design'*Z*W*Z'*delta;
    beta = coef(1:size(X,2));
    alpha0 = coef(length(coef));
end

% Calculate xi's
xi = delta - X*beta + alpha0*P;

% Construct the vector of estimated parameters
est = [alpha0;alpha1;beta];

% Return either the sum of moments or the moments for each observation
if ret_sum
    g = Z'*xi;
else
    g = diag(xi)*Z;
end