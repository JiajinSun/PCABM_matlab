classdef CAdcBlkMod
% dcBlkMod
    properties
        pri                % prior on community labels (pi)
        P                  % edge probability matrix (K*K)
        n                  % number of nodes
        K                  % number of communities
        %lambda             % theoretical expected node degree
                            % average degree = lambda * E(exp(Z^T gamma))
        rhon               % \rho_n: P = \rho_n \bar(P), \bar(P) fixed
                            % rho -> 0, n^2 * rho -> \infty
        
%         isdc               % a logical, ==1 means dc, ==0 when LowVal = 1 or LowProb = 0     
        thv                % a vector thetas for degree-correction
        thp                % probability of each value in thv
        exTheta            % expected vaue of theta
        
        
        cvt                % n*n*p covariate matrix
        gamma              % reg parameter of covariate (p-dim vector)
        
        theta              % the actual theta being used
        c                  % the actual labels being used
        As                 % the actual adjacency matrix being used
    end
    
    methods
        function ob = CAdcBlkMod(n,K, LowVal, LowProb, rho, pri)
            %LowProb = 0: no degree correction
            ob.K = K;
            %ob.lambda = lambda;
            ob.n = n;
            
            if nargin <= 5
                pri = 1/K * ones(1,K) ;
            end
            
            ob.pri = pri;
            ob.thv = [1 LowVal];
            ob.thp = [1-LowProb LowProb];
            
            ob.exTheta = ob.thp(:)' * ob.thv(:);
            
%             ob.cvt = cvt;
%             ob.gamma = gamma;
            ob.rhon = rho;
        end
        
        function ob = genP(ob, OIR, inWeights)
            % Generate P matrix with OIR and inWeights
            
            diag_idx = diag(true(ob.K,1));
            
            if OIR == 0
                ob.P = diag(ones(ob.K,1));
            else
                ob.P = ones(ob.K);
                ob.P( diag_idx ) = ob.P(1,2)/OIR;
            end

            ob.P(diag_idx) = inWeights(:).*ob.P(diag_idx);
            
            ob.P = ob.P * ob.rhon;

            %tmplam = (ob.n-1) * ob.pri(:)' * ob.P * ob.pri(:) * (ob.exTheta)^2;
%             tmplam = dcBlkMod.compLam(ob.n, ob.pri, ob.P, ob.exTheta);
%             ob.P = ob.P *(ob.lambda/tmplam);
            
%             if max(ob.P(:)) > 1
%                 warning('overflow:Pmat','Maximum of P matrix is above 1.')
%             end
        end
        
        function ob = CAgen_latent(ob)
            [ob.c,~] = find(mnrnd(1,ob.pri,ob.n)');
% ob.c= ones(ob.n,1);
% ob.c(1:ob.n/2) = 2;   %%%% experiment

            ob.theta = randsrc(ob.n,1,[ob.thv; ob.thp]);
        end
        
        
        function ob = CAgenData(ob, cvt, gamma)
%             
%             [ob.c,~] = find(mnrnd(1,ob.pri,ob.n)');
%             ob.theta = randsrc(ob.n,1,[ob.thv; ob.thp]);
            ob.cvt = cvt;
            ob.gamma = gamma;
        
%             ob.As = CAgenCAdcBM0(ob.c, ob.P, ob.theta, ob.cvt, ob.gamma);
             ob.As = CAgenCAdcBM1(ob.c, ob.P, ob.theta, ob.cvt, ob.gamma);
% ob.As = CAgenCAdcBM_mean(ob.c, ob.P, ob.theta, ob.cvt, ob.gamma); %%% experiment: mean

         
        end
        
        function saveToFile(ob,tag)
            
            if nargin < 2
                tag = '';
            end
            dlmwrite(['graph' tag '.txt'], full(ob.As))
            dlmwrite(['clusters' tag '.txt'], ob.c)
            %dlmwrite(['thetas' tag '.txt'], ob.theta)
          
        end
        
        function ob = perturbData(ob,rho)
            Bs = genBlkMod(ones(ob.n,1),log(ob.n)/ob.n,log(ob.n)*ob.n); %where is this func?
            Bs = Bs + Bs';

            avgDeg = full(mean(sum(ob.As,2))); % average degree of As
            ob.As = ob.As + Bs * rho * avgDeg/log(ob.n);
        end
        
        
        function ob = removeZeroDeg(ob)
            zdNodes = sum(ob.As,2) == 0;
%             nOrg = size(ob.As,1);
            

           % ob.n = ob.n - sum(zdNodes);
            ob.As = ob.As(~zdNodes,~zdNodes);
            ob.c = ob.c(~zdNodes); 
            ob.theta = ob.theta(~zdNodes); 
            ob.cvt = ob.cvt(~zdNodes,~zdNodes,:);
        end
        
  
    end
    
      
    methods(Static)
        function lam = compLam(n, pri, P, exTheta)
            lam = (n-1) * pri(:)' * P * pri(:) * (exTheta)^2;
        end
    end
end