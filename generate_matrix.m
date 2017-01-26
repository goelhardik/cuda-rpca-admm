function generate_matrix(m, n, r)
s = RandStream.create('mt19937ar','seed',5489);
RandStream.setGlobalStream(s);

N = 3;

L = randn(m,r) * randn(r,n);    % low rank
S = sprandn(m,n,0.05);          % sparse
S(S ~= 0) = 20*binornd(1,0.5,nnz(S),1)-10;
V = 0.01*randn(m,n);            % noise

A = S + L + V;
A = single(A);

[m, n] = size(A)

fid = fopen([int2str(n) 'A.dat'],'w');
fwrite(fid,[m,n],'int');
fwrite(fid,A,'float');
fclose(fid);
end
