function [e] = init_e_gen(c,init_accu)
% generate an initial class assignment e that has accuracy 'init_accu' given
% true class assignment 'c'
n = length(c);
k = round(n * (1-init_accu));
e0 = zeros(n,1);
e0(randsample(n,k))=1;
e=abs(c - 1 - e0)+1;
end

