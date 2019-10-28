function [delta1] = contr_map(alpha1, delta0, S, P, LnYSim, tol)
% Solve the system s(theta,delta) = s_observed for
% delta at a given theta through BLP's contraction
% mapping.

% Initialize values
delta1 = delta0;
change = 1e10;

while change > tol 
    
    % Compute implied shares s(theta,delta0)
    S_hat = comp_shares(alpha1, delta0, P, LnYSim);
    
    % Update delta comparing with observed shares
    delta1 = delta0 + (log(S)-log(S_hat));
    
    % Update solution and exit criterion
    change = norm(delta1 - delta0);
    delta0 = delta1;
    
end

