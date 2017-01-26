function admm_example(filename)

fid = fopen(filename,'r');
m = fread(fid,1,'int');
n = fread(fid,1,'int');
A = fread(fid,[m, n],'float');
fclose(fid);
A = single(A);

[m, n] = size(A);
g2_max = norm(A(:),inf);
g3_max = norm(A);
g2 = 0.15*g2_max;
g3 = 0.15*g3_max;

K = 3;
h3 = admm(A);
[m, n] = size(A);

fprintf('Writing output matrices in boyd_*.dat\n');
fid = fopen([int2str(n) 'boyd_X1.dat'],'w');
fwrite(fid,h3.X1_admm','float');
fclose(fid);

fid = fopen([int2str(n) 'boyd_X2.dat'],'w');
fwrite(fid,h3.X2_admm','float');
fclose(fid);

fid = fopen([int2str(n) 'boyd_X3.dat'],'w');
fwrite(fid,h3.X3_admm','float');
fclose(fid);
