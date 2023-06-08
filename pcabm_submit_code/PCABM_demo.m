addpath(fullfile('.','cpl_m_code'))


    n = 200;
    K = 2;    % number of communities
    oir = 0.5;    % "O"ut-"I"n-"R"atio 

    compErr = @(c,e) compMuI(compCM(c,e,K));    % use mutual info as a measure of error/sim.
    compErr2 = @(c,e) compARI(compCM(c,e,K));   % adjusted rand index

    inWei = ones(1,K);  % relative weight of in-class probabilities
    % theta:
    % [lowVal, lowProb] = deal(0.2, 0.9); % An example of Degree corrected block model 
     [lowVal, lowProb] = deal(1,0); % An example of original block model

    % settings of rho and gamma
    p = 5;
    c_rho = 5;
    c_gam = 1.2;
    rho = c_rho * log(n) / n;
   
    gamma = (0.4:0.4:2)' * c_gam;
    
    fprintf("n = %1.1f, \n c_rho = %2.2f, c_gam = %2.2f, K  = %1.1f, \n oir = %2.2f",n,c_rho,c_gam,K, oir) 
    
    NUM_rep = 100;
    init_nmi = zeros(NUM_rep,1);
    init_ari = zeros(NUM_rep,1);   %saves results for scwa
    CAplEM_nmi = zeros(NUM_rep,1);
    CAplEM_ari = zeros(NUM_rep,1); %saves results for plem
    gammahat = zeros(NUM_rep,p);   %saves the gammahat

for rep = 1:NUM_rep
    
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


    %%%% spectral clustering:
        %%%%% perturb = false: no regularization;
        %%%%% perturb = true, SC_r = false: regularization of the form in Amini et al., 2013;
        %%%%% perturb = true, SC_r = true: regularization of the form in the PCABM paper Huang et al., 2023+;
    init_opts = struct('verbose',false,'perturb',true,...
                   'score',false,'divcvt',true,'D12',false,'SC_r',true);
%     tic, fprintf('%-40s','Applying init. method (SC) ...') 

    [e, init_dT] = CA_SCWA(mo.As, mo.K, mo.cvt, gammah, init_opts);    
%     fprintf('%3.5fs\n',toc)
    init_nmi = compErr(mo.c, e);
    init_ari = compErr2(mo.c, e);


    %%%% PLEM for covariate adjusted
    T = 20;
    cpl_opts = struct('verbose',false,'delta_max',0.1, ...
                       'itr_num',T,'em_max',80,'track_err',true);
%     tic, fprintf('%-40s','Applying CA plEM ...') 
    [CA_chat, CA_err, CA_dT, CA_post] = ...
         CA_plEM(mo.As, mo.K, e, mo.c, mo.cvt, gammah , cpl_opts);
%     fprintf('%3.5fs\n',toc)
    CAplEM_nmi = compErr(mo.c, CA_chat);
    CAplEM_ari = compErr2(mo.c, CA_chat);
end

result_gamma = [mean(gammahat, 1),var(gammahat, 1)];
result_err = mean([init_nmi,init_ari,CAplEM_nmi,CAplEM_ari],1);
result_std = sqrt([var(init_nmi,1),var(init_ari,1),var(CAplEM_nmi,1),var(CAplEM_ari,1)]);
result_gamma
result_err
result_std

% result500_gamma = mean(gammahat, 1);
% result500_err = mean([init_nmi,init_ari,CAplEM_nmi,CAplEM_ari,nCAplEM_nmi,nCAplEM_ari] , 1);
% result400_gamma = mean(gammahat, 1);
% result400_err = mean([init_nmi,init_ari,CAplEM_nmi,CAplEM_ari] , 1);
% result300_gamma = mean(gammahat, 1);
% result300_err = mean([init_nmi,init_ari,CAplEM_nmi,CAplEM_ari] , 1);
% result200_gamma = mean(gammahat, 1);
% result200_err = mean([init_nmi,init_ari,CAplEM_nmi,CAplEM_ari] , 1);
% result100_gamma = mean(gammahat, 1);
% result100_err = mean([init_nmi,init_ari,CAplEM_nmi,CAplEM_ari] , 1);

% figure(1), clf, hold on
% plot(100:100:500,...
%     [result100_err(2),result200_err(2),result300_err(2),result400_err(2),result500_err(2)],...
%     'marker','+', 'Color', 'b')
% plot(100:100:500,...
%     [result100_err(4),result200_err(4),result300_err(4),result400_err(4),result500_err(4)],...
%     'marker','o', 'Color', 'r')
% legend('scwa', 'CAplEM')
% xlabel('n')
% ylabel('ARI')

