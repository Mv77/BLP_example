%% Data processing

% Read data
Data = readtable('../Data/Data.txt');

% Add an intercept
Data.cons = ones(size(Data,1),1);

% Rescale price, weight and hp
Data.p = Data.p * 1e-3;
Data.weight = Data.weight * 1e-3;
Data.hp = Data.hp * 1e-2;

% Create variable names and convenient grouped variables as matrices
X_names = {'weight','cons','hp','ac'};
P_names = {'p'};
Q_names = {'q'};
Id_names = {'firm_id'};

X = table2array(Data(:,X_names));
P = table2array(Data(:,P_names));
Q = table2array(Data(:,Q_names));
Id = table2array(Data(:,Id_names));

% Instruments (placeholder)
Z = table2array(Data(:,'weight'))+normrnd(3,1,[size(X,1),1]);

%% Create BLP instruments

% I use the characteristics-based instruments. So I first define which
% characteristics I will use.
charact = {'weight','hp','ac'};
Nchar = length(charact);

% For each of them I will add to the set of instruments:
% - The characteristic itself. (x)
% - The sum of the characteristic for other products of the same firm. (x_of)
% - The sum of the characteristic for products of all other firms. (x_ef)

% Initialize empty istrument matrix and list of names.
Z_names = cell(1,Nchar);
Z = zeros(length(P),Nchar*3);

% Go through characteristics
for k = 1:Nchar
    
    ind = (k-1)*Nchar;
    x = table2array(Data(:,charact{k}));
    
    % 1. Characteristic itself
    Z(:,ind+1) = x;
    Z_names{ind+1} = charact{k};
    
    % 2. Sum for other products of the same firm.
    sums = accumarray(Id,x);
    Z(:,ind+2) = sums(Id) - x;
    Z_names{ind+2} = strcat(charact{k},'_of');
    
    % 3. Sum for products of all other firms
    total = repmat(sum(x),[length(P),1]);
    Z(:,ind+3) = total - sums(Id);
    Z_names{ind+3} = strcat(charact{k},'_ef');
    
end

clear ind k sums total