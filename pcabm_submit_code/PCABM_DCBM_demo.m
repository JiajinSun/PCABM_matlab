addpath(fullfile('.','cpl_m_code'))


    n = 200;
    K = 2;    % number of communities
    oir = 0.5;    % "O"ut-"I"n-"R"atio 

    compErr = @(c,e) compMuI(compCM(c,e,K));    % use mutual info as a measure of error/sim.
    compErr2 = @(c,e) compARI(compCM(c,e,K));   % adjusted rand index

    inWei = ones(1,K);   % relative weight of in-class probabilities 
    % degree correction parameters:
     [lowVal, lowProb] = deal(4, 0.5); % An example of Degree corrected block model
%      [lowVal, lowProb] = deal(1,0); % An example of original block model


     p = 1;
    c_rho = 3;
%     c_gamma = 1.5;
    rho = c_rho * (log(n)) / n ;

   
    
    NUM_rep = 100;
    init_nmi = zeros(NUM_rep,1);
    init_ari = zeros(NUM_rep,1);
 %   SCWA_accu = zeros(NUM_rep,1);
    CAplEM_nmi = zeros(NUM_rep,1);
    CAplEM_ari = zeros(NUM_rep,1);
 %   CAplEM_accu = zeros(NUM_rep,1);
    gammahat = zeros(NUM_rep,p);
    
    
cvt = zeros(n,n,1);
gamma = 0;

for rep = 1:NUM_rep



    mo = CAdcBlkMod(n,K, lowVal, lowProb, rho); % create a base model
    mo = mo.genP(oir, inWei);  % generate the edge probability matrix


%     tic, fprintf('%-40s','Generating data ...')
    mo = mo.CAgen_latent;
    mo = mo.CAgenData(cvt,gamma);        % generate data (Adj. matrix "As" and the labels "c")
    mo = mo.removeZeroDeg;  % remove zero degree nodes
    nnz = size(mo.As, 1);
%     fprintf('%3.5fs\n',toc)

cvtD = zeros(nnz,nnz,p);
cvtDup = triu( log( repmat(sum(mo.As,2),1,nnz) ) , 1 ) + triu( log( repmat(sum(mo.As,1),nnz,1) ) , 1 );
cvtD(:,:,1) = cvtDup + cvtDup';

%     tic, fprintf('%-40s','Estimating gamma ...')
    pi0 = ones(K,1) / K;
    [e0,~] = find(mnrnd(1,pi0,nnz)');
    gammah = CA_estimgamma(mo.As, K, e0, cvtD);
%     gammah2 = CA_estimgamma2(mo.As, K, e0, mo.cvt); 
%     fprintf('%3.5fs\n',toc)
    gammahat(rep,:) = gammah;




    %%%% spectral clustering:
        %%%%% perturb = false: no regularization;
        %%%%% perturb = true, SC_r = false: regularization of the form in Amini et al., 2013;
        %%%%% perturb = true, SC_r = true: regularization of the form in the PCABM paper Huang et al., 2023+;
    init_opts = struct('verbose',false,'perturb',true,...
                   'score',false,'divcvt',true,'D12',false,'SC_r',true);
    T = 20;
%     tic, fprintf('%-40s','Applying init. method (SC) ...') 

    [e, init_dT] = CA_SCWA(mo.As, mo.K, cvtD, gammah, init_opts);    
%     fprintf('%3.5fs\n',toc)
    init_nmi(rep) = compErr(mo.c, e);
    init_ari(rep) = compErr2(mo.c, e);
 %   SCWA_accu_temp = sum(abs(e-mo.c))/n;
 %   SCWA_accu(rep) = max(SCWA_accu_temp, 1-SCWA_accu_temp);


    % the function of EM for covariate adjusted
    cpl_opts = struct('verbose',false,'delta_max',0.1, ...
                       'itr_num',T,'em_max',80,'track_err',true);
%     tic, fprintf('%-40s','Applying CA plEM ...') 
    [CA_chat, CA_err, CA_dT, CA_post] = ...
         CA_plEM(mo.As, mo.K, e, mo.c, cvtD, gammah , cpl_opts);
%     fprintf('%3.5fs\n',toc)
    CAplEM_nmi(rep) = compErr(mo.c, CA_chat);
    CAplEM_ari(rep) = compErr2(mo.c, CA_chat);
 %   CAplEM_accu_temp = sum(abs(CA_chat-mo.c))/n;
 %   CAplEM_accu(rep) = max(CAplEM_accu_temp, 1-CAplEM_accu_temp);
%     fprintf(1,'Init NMI = %3.2f\nCAplEM  NMI = %3.2f\n\n',init_nmi,CAplEM_nmi)

end

fprintf('n=%d, c_rho=%f, lowVal=%f, lowProb=%f\n', n, c_rho, lowVal, lowProb)
result_gamma = [mean(gammahat, 1),var(gammahat, 1)];
result_err = mean([init_nmi,init_ari,CAplEM_nmi,CAplEM_ari],1);
result_std = sqrt([var(init_nmi,1),var(init_ari,1),var(CAplEM_nmi,1),var(CAplEM_ari,1)]);
result_gamma
result_err
result_std

