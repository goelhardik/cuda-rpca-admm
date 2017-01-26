function h = admm(XO)

A = XO;
[m, n] = size(A);

g2_max = norm(A(:),inf);
g3_max = norm(A);
g2 = 0.15*g2_max;
g3 = 0.15*g3_max;

% ADMM


MAX_ITER = 200;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;


lambda = 1;
rho = 1/lambda;

N = 3;
X_1 = zeros(m,n);
X_2 = zeros(m,n);
X_3 = zeros(m,n);
z   = zeros(m,N*n);
U   = zeros(m,n);

fprintf('\n%3s\t%10s\t%10s\n', 'iter', ...
    'r norm', 's norm');

timerval = tic;
for k = 1:MAX_ITER

    B = avg(X_1, X_2, X_3) - A./N + U;
    % x-update
    X_1 = (1/(1+lambda))*(X_1 - B);
    X_2 = prox_l1(X_2 - B, lambda*g2);
    X_3 = prox_matrix(X_3 - B, lambda*g3, @prox_l1);

    % (for termination checks only)
    x = [X_1 X_2 X_3];
    zold = z;
    z = x + repmat(-avg(X_1, X_2, X_3) + A./N, 1, N);

    % u-update
    U = B;

    % diagnostics, reporting, termination checks
    %h.objval(k)   = objective(X_1, g2, X_2, g3, X_3);
    h.r_norm(k)   = norm(x - z,'fro');
    h.s_norm(k)   = norm(-rho*(z - zold),'fro');
    h.eps_pri(k)  = sqrt(m*n*N)*ABSTOL + RELTOL*max(norm(x,'fro'), norm(-z,'fro'));
    h.eps_dual(k) = sqrt(m*n*N)*ABSTOL + RELTOL*sqrt(N)*norm(rho*U,'fro');

        fprintf('%4d\t%10.4f\t%10.4f\t\n', k, ...
            h.r_norm(k), h.s_norm(k));

    if h.r_norm(k) < h.eps_pri(k) && h.s_norm(k) < h.eps_dual(k)
         break;
    end

end

h.admm_toc = toc(timerval);
h.admm_iter = k;
h.X1_admm = X_1;
h.X2_admm = X_2;
h.X3_admm = X_3;

fprintf('Time taken = %f\n', h.admm_toc);

end

function x = avg(varargin)
    N = length(varargin);
    x = 0;
    for k = 1:N
        x = x + varargin{k};
    end
    x = x/N;
end

function p = objective(X_1, g_2, X_2, g_3, X_3)
    p = norm(X_1,'fro').^2 + g_2*norm(X_2(:),1) + g_3*norm(svd(X_3),1);
end


function x = prox_l1(v, lambda)
% PROX_L1    The proximal operator of the l1 norm.
%
%   prox_l1(v,lambda) is the proximal operator of the l1 norm
%   with parameter lambda.

    x = max(0, v - lambda) - max(0, -v - lambda);
end

function x = prox_matrix(v, lambda, prox_f)
% PROX_MATRIX    The proximal operator of a matrix function.
    R = chol(transpose(v) * v); 
    Q = v* inv(R);
    [UR S V] = svd(R, 'econ');
    U = Q * UR; 
    x = U*diag(prox_f(diag(S), lambda))*V';
end
