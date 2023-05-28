addpath(fullfile('.','cpl_m_code'))

%     tic, fprintf('%-40s','Setting up the model ...')
    n = 200;
    K = 2;    % number of communities
    oir = 0.5;    % "O"ut-"I"n-"R"atio 

    compErr = @(c,e) compMuI(compCM(c,e,K));    % use mutual info as a measure of error/sim.
    compErr2 = @(c,e) compARI(compCM(c,e,K));   % adjusted rand index

    inWei = [1 1];   % relative weight of in-class probabilities
    % theta:
    % [lowVal, lowProb] = deal(0.2, 0.9); % An example of Degree corrected block model 
     [lowVal, lowProb] = deal(1,0); % An example of original block model

    p = 5;
%     p = 1;
    c_rho = 2;
    c_gam = 1.5;
    rho = c_rho * (log(n)) / n;
     init_accu = 0.6;  %%%%% initial accuracy, varies from 0.5 to 1, as in section 7.3 of the PCABM paper
   
        gamma = (0.4:0.4:2)' * c_gam;
%       gamma = 2.5;
    fprintf("n = %1.1f, \n c_rho = %2.2f, c_gam = %2.2f, K  = %1.1f, \n oir = %2.2f",n,c_rho,c_gam,K, oir) 

    NUM_rep = 100;
    init_nmi = zeros(NUM_rep,1);
    init_ari = zeros(NUM_rep,1);
    SCWA_accu = zeros(NUM_rep,1);
    CAplEM_nmi = zeros(NUM_rep,1);
    CAplEM_ari = zeros(NUM_rep,1);
    CAplEM_accu = zeros(NUM_rep,1);
    init_gen_accu = zeros(NUM_rep,1);
    gammahat = zeros(NUM_rep,p);

for rep = 1:NUM_rep
    
     cvt = zeros(n,n,p);
    cvtup = triu( binornd(1,0.1, n,n), 1 );
    cvt(:,:,1) = cvtup + cvtup';
%     cvtup = triu( binornd(1,0.1, n,n), 1 );
     cvtup = triu( poissrnd(0.1, n,n), 1 );
    cvt(:,:,2) = cvtup + cvtup';
    cvtup = triu( rand(n,n), 1 );
    cvt(:,:,3) = cvtup + cvtup';
%     cvtup = triu( binornd(1,0.1, n,n), 1 );
     cvtup = triu( exprnd(0.3,n,n), 1 );
    cvt(:,:,4) = cvtup + cvtup';
    cvtup = triu( randn(n,n), 1 ) * 0.3;
    cvt(:,:,5) = cvtup + cvtup';

% cvtup = triu( poissrnd(0.1, n,n), 1 );
% cvt(:,:,1) = cvtup + cvtup';
    

    mo = CAdcBlkMod(n,K, lowVal, lowProb, rho); % create a base model
    mo = mo.genP(oir, inWei);  % generate the edge probability matrix
%     fprintf('%3.5fs\n',toc)

%     tic, fprintf('%-40s','Generating data ...')
    mo = mo.CAgen_latent;
    mo = mo.CAgenData(cvt,gamma);        % generate data (Adj. matrix "As" and the labels "c")
    mo = mo.removeZeroDeg;  % remove zero degree nodes
    nnz = size(mo.As, 1);
%     fprintf('%3.5fs\n',toc)

%     tic, fprintf('%-40s','Estimating gamma ...')
    pi0 = ones(K,1) / K;
    [e0,~] = find(mnrnd(1,pi0,nnz)');
    gammah = CA_estimgamma(mo.As, K, e0, mo.cvt);
%     gammah2 = CA_estimgamma2(mo.As, K, e0, mo.cvt); 
%     fprintf('%3.5fs\n',toc)
    gammahat(rep,:) = gammah;

%     gammah = 3.5;

     T = 20;

    %%%% spectral clustering:
        %%%%% perturb = false: no regularization;
        %%%%% perturb = true, SC_r = false: regularization of the form in Amini et al., 2013;
        %%%%% perturb = true, SC_r = true: regularization of the form in the PCABM paper Huang et al., 2023+;
    init_opts = struct('verbose',false,'perturb',true,...
                   'score',false,'divcvt',true,'D12',false,'SC_r',true);
%     tic, fprintf('%-40s','Applying init. method (SC) ...') 

    [e, init_dT] = CA_SCWA(mo.As, mo.K, mo.cvt, gammah, init_opts);    
%     fprintf('%3.5fs\n',toc)
    init_nmi(rep) = compErr(mo.c, e);
    init_ari(rep) = compErr2(mo.c, e);
    SCWA_accu_temp = sum(abs(e-mo.c))/n;
    SCWA_accu(rep) = max(SCWA_accu_temp, 1-SCWA_accu_temp);

     %%%% certain initial accuracy
     e = init_e_gen(mo.c,init_accu);
     init_gen_accu_temp = sum(abs(e-mo.c))/n;
     init_gen_accu(rep) = max(init_gen_accu_temp, 1- init_gen_accu_temp);
 %     sum(abs(e-mo.c))


    % the function of EM for covariate adjusted
    cpl_opts = struct('verbose',false,'delta_max',0.1, ...
                       'itr_num',T,'em_max',80,'track_err',true);
%     tic, fprintf('%-40s','Applying CA plEM ...') 
    [CA_chat, CA_err, CA_dT, CA_post] = ...
         CA_plEM(mo.As, mo.K, e, mo.c, mo.cvt, gammah , cpl_opts);
%     fprintf('%3.5fs\n',toc)
    CAplEM_nmi(rep) = compErr(mo.c, CA_chat);
    CAplEM_ari(rep) = compErr2(mo.c, CA_chat);
    CAplEM_accu_temp = sum(abs(CA_chat-mo.c))/n;
    CAplEM_accu(rep) = max(CAplEM_accu_temp, 1-CAplEM_accu_temp);
%     fprintf(1,'Init NMI = %3.2f\nCAplEM  NMI = %3.2f\n\n',init_nmi,CAplEM_nmi)
  
% % compare with no covariate plEM
%     [nCA_chat,~, ~, ~] = ...
%          CA_plEM(mo.As, mo.K, e, mo.c, zeros(nnz,nnz,p), zeros(p,1) , cpl_opts);
%     nCAplEM_nmi = compErr(mo.c, nCA_chat);
%     nCAplEM_ari = compErr2(mo.c, nCA_chat);
end

result_gamma = [mean(gammahat, 1),var(gammahat, 1)];
result_err = mean([init_gen_accu,SCWA_accu,init_nmi,init_ari,CAplEM_accu,CAplEM_nmi,CAplEM_ari],1);
result_std = sqrt([var(init_gen_accu,1),var(SCWA_accu,1),var(init_nmi,1),var(init_ari,1),var(CAplEM_accu,1),var(CAplEM_nmi,1),var(CAplEM_ari,1)]);
result_gamma
result_err
result_std

