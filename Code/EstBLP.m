%% Preamble

% Set seed
rng(123);

% split_par defines whether to perform the parameter search
% only over the "non-linear" parameters, as suggested in
% Nevo (2000).
% If set to false, the search is performed over all parameters.
split_par = false;

%% Optimization setup
run('Optimizer_options.m')

%% Data processing
run('ReadData.m')

% Define N
N = 1e8;
% Compute shares
S = Q/N;

% Since we have a single market, the own-firm sum of charact.
% and all other firms' sum of charact. instruments are almost
% perfectly correlated. We therefore exclude the latter.
Z = Z(:,[1,2,4,5,7,8]);

%% Simulate log-income
Nsim = 1e3;
mu = 10.25;
sig = 0.8;
LnYSim = normrnd(mu, sig, [Nsim,1]);

%% Initial params

% Starting values
alpha_0 = 0;
alpha_1 = 0;
beta_cons = 1;
beta_weight = 1;
beta_hp = 0;
beta_ac = 0;

if split_par
    p0 = alpha_1;
else
    p0 = [alpha_0, alpha_1, beta_cons, beta_weight, beta_hp, beta_ac]';
end

% Initial weight matrix
W_0 = eye(size(Z,2));

%% Initial estimator to obtain efficient weight matrix

% Loss function minimization:

% Initial derivative-free search
opt_0 = fminsearch(@(p) GMMloss(p,S,X,P,LnYSim,Z,W_0,split_par),...
                   p0,fminsearch_options);
               
% Try to improve using derivative-based mathod.
opt_0 = fminunc(@(p) GMMloss(p,S,X,P,LnYSim,Z,W_0,split_par),...
                opt_0,fminunc_options);
            
% Get the moments at every observation
g = moment(opt_0,S,X,P,LnYSim,Z,W_0,false, split_par);

% Compute W_hat from "Econometrics" Hansen (2018), Chap 13.25.
W_hat = zeros(size(Z,2));
g_bar = mean(g,1)';
for i = 1:size(Z,1)
    W_hat = W_hat + g(i,:)'*g(i,:);
end
W_hat = inv(W_hat/size(Z,2) - g_bar*g_bar');

%% Estimation

% Re-optimize with the new weight matrix and starting from the previously
% found point
opt = fminsearch(@(p) GMMloss(p,S,X,P,LnYSim,Z,W_hat,split_par),...
                 opt_0,fminsearch_options);

% Once a good solution is found, try to improve it with a derivative-based
% method
[opt, fval] = fminunc(@(p) GMMloss(p,S,X,P,LnYSim,Z,W_hat,split_par),...
                      opt,fminunc_options);
                    
% Evaluate again just to get full estimates (in case the search was
% done only over non-linear parameters)
[~, est] = GMMloss(opt,S,X,P,LnYSim,Z, W_hat, split_par);

% Variance-covariance and standard error estimation
vcov = avar_est(est,S,X,P,LnYSim,Z, W_hat)/size(P,1);
se = sqrt(diag(vcov));

%% Output

% List names of the parameters
par_names = {'$\alpha_0$','$\alpha_1$','$\beta_{cons}$',...
             '$\beta_{weight}$','$\beta_{hp}$','$\beta_{ac}$'};
         
% Format estimates in a table
Estimates = table(est,se,...
    'VariableNames',{'Estimate','S.E.'},...
    'RowNames',par_names);

% Add objective funtion (and an empty line to separate)
Estimates('\,',:) = {NaN,NaN};
Estimates('\textbf{F.Obj}',:) = {fval,NaN};

Estimates