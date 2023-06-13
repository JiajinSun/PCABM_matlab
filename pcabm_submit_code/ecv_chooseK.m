addpath(fullfile('.','cpl_m_code'))

rng(2000);


n = 1000;
K = 2;    % number of communities
% oir = 1 / (K/2+1);    % "O"ut-"I"n-"R"atio 
oir = 0.5;    % "O"ut-"I"n-"R"atio. oir = 0.5 means $\bar B$ has diagonal = 2, off-diagonal = 1. 

compErr = @(c,e) compMuI(compCM(c,e,K));   % calculate mutual info as a measure of error/sim.
compErr2 = @(c,e) compARI(compCM(c,e,K));   % calculate adjusted rand index as a measure of error/sim.

inWei = ones(1,K);  % relative weight of in-class probabilities
%   [lowVal, lowProb] = deal(0.2, 0.9); % An example of Degree corrected block model %lowProb = rho
  [lowVal, lowProb] = deal(1,0); % An example of original block model

c_rho = 5;
rho = c_rho * (log(n)) / n;   
p = 5; %dim of covariate Z
c_gam = 1;
gamma = (0.4:0.4:2)' * c_gam ;

fprintf("c_rho = %2.2f, c_gam = %2.2f, K  = %1.1f \n, oir = %2.2f",c_rho,c_gam,K, oir)

experiment_times = 100;

Khat_lik_scaled = zeros(experiment_times,1);
Khat_se = zeros(experiment_times,1);
for realization = 1:experiment_times                       
cvt = zeros(n,n,p);
    cvtup = triu( binornd(1,0.1, n,n), 1 );
    cvt(:,:,1) = cvtup + cvtup';
    cvtup = triu( poissrnd(0.1, n,n), 1 );
    cvt(:,:,2) = cvtup + cvtup';
    cvtup = triu( rand(n,n), 1 );
    cvt(:,:,3) = cvtup + cvtup';
    cvtup = triu( exprnd(0.3,n,n), 1 );
    cvt(:,:,4) = cvtup + cvtup';
    cvtup = triu( randn(n,n), 1 ) * 0.3;
    cvt(:,:,5) = cvtup + cvtup';

mo = CAdcBlkMod(n,K, lowVal, lowProb, rho); % create a base model
mo = mo.genP(oir, inWei);  % generate the edge probability matrix
% fprintf('%3.5fs\n',toc)

% tic, fprintf('%-40s','Generating data ...')
mo = mo.CAgen_latent;

mo = mo.CAgenData(cvt, gamma);        % generate data (Adj. matrix "As" and the labels "c")
mo = mo.removeZeroDeg;  % remove zero degree nodes
nnz = size(mo.As, 1);
% fprintf('%3.5fs\n',toc)

% tic, fprintf('%-40s','Estimating gamma with K=1 ...')
% pi0 = ones(K,1) / K;
% [e0,~] = find(mnrnd(1,pi0,nnz)');
e0 = ones(nnz,1);
 gammah = CA_estimgamma(mo.As, 1, e0, mo.cvt);
% fprintf('%3.5fs\n',toc)

%%%%%% ecv for selecting K
    expcvt =exp( reshape( (reshape(mo.cvt, nnz*nnz, p) * gammah), nnz, nnz)); %n*n matrix of exp(Zij^T gammah)
    A1 = mo.As ./ expcvt;
Nrep = 5; %number of folds
Kmax = 6;
LKm_lik = zeros(Kmax,Nrep);
LKm_lik_scaled = zeros(Kmax,Nrep);
LKm_se = zeros(Kmax,Nrep);
for m=1:Nrep % cross validation; m-th fold
    p_subsam = 0.9;
    subOmega = binornd(1,p_subsam,nnz,nnz);
    subOmega = triu(subOmega) + triu(subOmega)'; %make it symmetric
    subsam_A1 = A1 .* subOmega;
    subsam_As = mo.As .* subOmega;
    subsam_expcvt = expcvt .* subOmega;
    [U,S,V] = svds(subsam_A1 / p, Kmax);
    for k = 1:Kmax % fit different models in each fold
        if sum(isnan(diag(S(1:k,1:k)) )) > 0 % if there is nan in the top 6 eigenvalues, 
                                             %  we think this k is wrong
            EA_hat = mo.As;
            % LKm_lik(k,m) = sum(sum( (mo.As-subsam_As) .* log(EA_hat) - EA_hat .* (1-subOmega) * 10 ));
            LKm_lik_scaled(k,m) = sum(sum( (A1-subsam_A1).* log(EA_hat) - EA_hat ./ expcvt .* (1-subOmega) * 10 ));   % scaled negative log-likelihood loss (snll)
            LKm_se(k,m) = sum(sum( ( ( A1 ) .* (1-subOmega) ).^2 )) * 10;   % scaled L2 loss
        else
        Ahat_k = U(:,1:k) * S(1:k,1:k)  * V(:,1:k)';
        
    %%% use the $A_ij * sqrt{lambda_i lambda_j}$ regularization as in PCABM paper
       opt_cvsc = struct('verbose',false,'perturb',true,...
                   'score',false,'divcvt',false,'D12',false,'SC_r',true); 
        eKm =  CA_SCWA(Ahat_k, k, zeros(nnz,nnz,1), 0, opt_cvsc);
            Oll = zeros(k);
            Ell = zeros(k);
            Bll = zeros(k);
            for ell1 = 1:k
                    for ell2 = 1:k
                    Oll(ell1,ell2) = sum( reshape( subsam_As(eKm==ell1, eKm==ell2), [], 1));
                    Ell(ell1,ell2) = sum( reshape( subsam_expcvt(eKm==ell1, eKm==ell2), [], 1));
                    Bll(ell1,ell2) = Oll(ell1,ell2) / Ell(ell1,ell2);
                    end
            end
          
            EA_hat = Bll(eKm,eKm) .* expcvt;
        
        LKm_lik_scaled(k,m) = sum(sum( (A1-subsam_A1).* log(EA_hat) - Bll(eKm,eKm) .* (1-subOmega) ));   % scaled negative log-likelihood loss (snll)
        LKm_se(k,m) = sum(sum( ( ( A1-Bll(eKm,eKm) ) .* (1-subOmega) ).^2 ));   % scaled L2 loss
        end
     end
end

LK_lik_scaled = mean(LKm_lik_scaled,2)/n;
LK_se = mean(LKm_se,2)/n;

[~,Khat_lik_scaled(realization)] = max(LK_lik_scaled);
[~,Khat_se(realization)] = min(LK_se);
end


fprintf("\n Nrep  = %1.1f \n",Nrep)
%%%% producing Table 2, for a particular $K$. 
%%%% the 1st output tabulate is for snll loss and 2nd is for scaled L2 loss.
%%%% each tabulate shows the frequency of $\hat K$. "value" means the value of $\hat K$.

tabulate(Khat_lik_scaled)
tabulate(Khat_se)
