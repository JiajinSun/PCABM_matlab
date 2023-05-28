addpath(fullfile('.','cpl_m_code'))

rng(2000);

% tic, fprintf('%-40s','Setting up the model ...')

n = 1000;
K = 2;    % number of communities
% oir = 1 / (K/2+1);    % "O"ut-"I"n-"R"atio 
oir = 1 / 2;    % "O"ut-"I"n-"R"atio. oir = 0.5 means $\bar B$ has diagonal = 2, off-diagonal = 1. 

compErr = @(c,e) compMuI(compCM(c,e,K));   % calculate mutual info as a measure of error/sim.
compErr2 = @(c,e) compARI(compCM(c,e,K));   % calculate adjusted rand index as a measure of error/sim.

%inWei = [1 1];   % relative weight of in-class probabilities
inWei = ones(1,K);
%   [lowVal, lowProb] = deal(0.2, 0.9); % An example of Degree corrected block model %lowProb = rho
  [lowVal, lowProb] = deal(1,0); % An example of original block model

c_rho = 5;
rho = c_rho * (log(n)) / n;   %dc+(log n)^1.5/n: totally fail: with pertb/score, 0.45; wt pertb, 0
                          %no dc, (log n)^1.5/n: ari = nmi = 1
p = 5; %dim of covariate Z
c_gam = 1;
gamma = (0.4:0.4:2)' * c_gam ;

fprintf("c_rho = %2.2f, c_gam = %2.2f, K  = %1.1f \n, oir = %2.2f",c_rho,c_gam,K, oir)

experiment_times = 100;
Khat_lik = zeros(experiment_times,1);
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
% gammah = gamma;
% fprintf('%3.5fs\n',toc)

%%%%%% ecv for selecting K
    expcvt =exp( reshape( (reshape(mo.cvt, nnz*nnz, p) * gammah), nnz, nnz)); %n*n matrix of exp(Zij^T gammah)
    A1 = mo.As ./ expcvt;
Nrep = 5; %number of folds
Kmax = 6;
LKm_lik = zeros(Kmax,Nrep);
LKm_lik_scaled = zeros(Kmax,Nrep);
LKm_se = zeros(Kmax,Nrep);
for m=1:Nrep
    p_subsam = 0.9;
    subOmega = binornd(1,p_subsam,nnz,nnz);
    subsam_A1 = A1 .* subOmega;
    subsam_As = mo.As .* subOmega;
    subsam_expcvt = expcvt .* subOmega;
    [U,S,V] = svds(subsam_A1 / p, Kmax);
    for k = 1:Kmax
        if sum(isnan(diag(S(1:k,1:k)) )) > 0 % this k is wrong
            EA_hat = mo.As;
            LKm_lik(k,m) = sum(sum( (mo.As-subsam_As) .* log(EA_hat) - EA_hat .* (1-subOmega) * 10 ));
            LKm_lik_scaled(k,m) = sum(sum( (A1-subsam_A1).* log(EA_hat) - EA_hat ./ expcvt .* (1-subOmega) * 10 ));   % scaled negative log-likelihood loss (snll)
            LKm_se(k,m) = sum(sum( ( ( A1 ) .* (1-subOmega) ).^2 )) * 10;   % scaled L2 loss
        else
        Ahat_k = U(:,1:k) * S(1:k,1:k)  * V(:,1:k)';
        
       % opt_cvsc = struct('verbose',false,'perturb',true,...
       %             'score',false,'divcvt',false,'D12',false);
    %%% use the $A_ij \sqrt{lambda_i lambda_j}$ regularization as in PCABM paper
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
        LKm_lik(k,m) = sum(sum( (mo.As-subsam_As) .* log(EA_hat) - EA_hat .* (1-subOmega) ));
        LKm_lik_scaled(k,m) = sum(sum( (A1-subsam_A1).* log(EA_hat) - Bll(eKm,eKm) .* (1-subOmega) ));   % scaled negative log-likelihood loss (snll)
        LKm_se(k,m) = sum(sum( ( ( A1-Bll(eKm,eKm) ) .* (1-subOmega) ).^2 ));   % scaled L2 loss
        end
     end
end
LK_lik = mean(LKm_lik,2)/n;
LK_lik_scaled = mean(LKm_lik_scaled,2)/n;
LK_se = mean(LKm_se,2)/n;
[~,Khat_lik(realization)] = max(LK_lik);
[~,Khat_lik_scaled(realization)] = max(LK_lik_scaled);
[~,Khat_se(realization)] = min(LK_se);
end

LK_lik
LK_lik_scaled
LK_se

fprintf("\n Nrep  = %1.1f \n",Nrep)
%%%% producing Table 2, for a particular $K$. 
%%%% the 2nd output tabulate is for snll loss and 3rd is for scaled L2 loss.
%%%% each tabulate shows the frequency of $\hat K$. "value" means the value of $\hat K$.
tabulate(Khat_lik)
tabulate(Khat_lik_scaled)
tabulate(Khat_se)

%%%%%%%% how different K affects gamma estimation?
% Kmax = 6;
% gammahk = zeros(p,Kmax);
% tic, fprintf('Estimating gamma for k = 1:%d ...',Kmax)
% for k = 1:Kmax
%     pi0k = ones(k,1) / k;
% [e0k,~] = find(mnrnd(1,pi0k,nnz)');
% gammahk(:,k) = CA_estimgamma(mo.As, k, e0k, mo.cvt);
% end
% fprintf('%\n 3.5fs\n',toc)
% err_gamk = sum( (gammahk - repmat(gamma, 1, Kmax)).^2, 1)
%different k almost does not affect gamma estimation; at least the true K
%has no advantage against other Ks. Thus, it is reasonable to use the same
%gamma for choosing Ks since its not favorable for the true K.

%%%%%%%% how subsampling affects gamma estimation?
% p_subsam = 0.9;
% subOmega = binornd(1,p_subsam,nnz,nnz);
% tic, fprintf('%-40s','Estimating gamma under ecv ...')
% pi0 = ones(K,1) / K;
% [e0,~] = find(mnrnd(1,pi0,nnz)');
% gammah_ecv = CA_estimgamma_ecv(mo.As, K, e0, mo.cvt, subOmega);
% fprintf('%3.5fs\n',toc)
%almost no difference with full data gamma estimation; at least it's hard to
%say which estimate is significantly better

%%%%%% use spectral clustering (w/wt pertubation) to initialize labels
% % init_opts = struct('verbose',false,'perturb',true,...
% %                     'score',true,'divcvt',false,'D12',false);
%  init_opts = struct('verbose',false,'perturb',true);
% T = 20;
% 
% tic, fprintf('%-40s','Applying init. method (SC) ...') 
% 
% [e, init_dT] = CA_SCWA(mo.As, mo.K, mo.cvt, gammah, init_opts);    
% fprintf('%3.5fs\n',toc)
% init_nmi = compErr(mo.c, e);
% init_ari = compErr2(mo.c, e);
% 
% 
% %%%%%% the function of EM for covariate adjusted
% cpl_opts = struct('verbose',false,'delta_max',0.1, ...
%                    'itr_num',T,'em_max',80,'track_err',true);
% tic, fprintf('%-40s','Applying CA plEM ...') 
% [CA_chat, CA_err, CA_dT, CA_post] = ...
%      CA_plEM(mo.As, mo.K, e, mo.c, mo.cvt, gammah , cpl_opts);
% fprintf('%3.5fs\n',toc)
% CAplEM_nmi = compErr(mo.c, CA_chat);
% CAplEM_ari = compErr2(mo.c, CA_chat);
% fprintf(1,'Init NMI = %3.2f\nCAplEM  NMI = %3.2f\n\n',init_nmi,CAplEM_nmi)


 
