function [s] = comp_shares(alpha1, delta, P, LnYSim)
% Computes market shares given parameters, deltas,
% and simulated individual characteristics

% Construct the individually varying-part
mu = - alpha1*LnYSim*P';

% Calculate utilities (without error)
% Columns are products, rows are individuals.
score = ones(size(LnYSim,1),1)*delta' + mu;
score = exp(score);

% Find individual purchase probabilities
denoms = 1 + sum(score,2);
S = score ./ repmat(denoms,[1, size(score,2)]);

% Shares are mean probabilities of purchase (accross individuals)
s = mean(S,1)';