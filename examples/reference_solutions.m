clear
clc
matrix_name = "Dubcova3";
matrix_path = sprintf("matrices/moderate/%s.mtx", matrix_name);
input_path = sprintf("matrices/testing/%s/input%d.mtx", matrix_name);
output_path = sprintf("matrices/testing/%s/output%d.mtx", matrix_name);

addpath("matrices/testing")
addpath("iluk")
A0 = mmread(matrix_path);

err = @(x,xref) norm(x-xref)/norm(xref);

[m, n] = size(A0);
xref = ones(m, 1);
b0 = A0*xref;
%% GMRES
% [L, U] = ilu(A0);
% xgmres = gmres(A0, b0, 100, 1e-13, 100);
% fprintf("Error GMRES=%.2e\n", err(xgmres,xref))
% fprintf("Error GMRES=%.2e   residual=%.2e\n\n", err(xgmres,x64), norm(b-A*xgmres)/norm(b))
%% double
% [m, n] = size(A0);
% xdouble = A0\b0;
%% With mp from Advanpix
% mp.Digits(32);
% tic
% A = mp(A0);
% b = mp(b0);
% x32 = A\b;
% mp.Digits(64);
% x64 = mp(A0)\mp(b0);
% toc

% fprintf("\nError double=%.2e   x32=%.2e\n", err(xdouble,x64), err(x32,x64))

%% VPA in Matlab
% A0 = double(A0);
% A0 = full(A0);
% digits(32);
% tic
% A = vpa(A0);
% b = vpa(b0);
% x32 = A\b;
% digits(64);
% A = vpa(A0);
% b = vpa(b0);
% x64 = A\b;
% toc
% fprintf("Error with VPA gmres=%.2e    double=%.2e   x32=%.2e\n", err(xgmres, x64), err(xdouble, x64), err(x32, x64))
% 
% writematrix(string(b), sprintf(input_path, 0), FileType="text")
% writematrix(string(x64), sprintf(output_path, 0), FileType="text");

% Visualize the sparsity pattern
% figure;
% spy(A0);
% title('Sparsity Pattern of Matrix A0');
% xlabel('Columns');
% ylabel('Rows');
% 
% A2 = A0 * A0;
% 
% figure;
% spy(A2);
% title('Sparsity Pattern of Matrix A2');
% xlabel('Columns');
% ylabel('Rows');
% 
% fprintf("A0 nnz = %d, A2 nnz = %d\n", nnz(A0), nnz(A2));

% A = full(A0);
% D = diag(1 ./ sqrt(diag(A)));
% A = D * A * D;
% 
% d = 0;
% for j = 1:n
%     for i = 1:j
%         if A(i, j) ~= 0
%             % fprintf("A(%d, %d) = %.6f\n", i-1, j-1, A(i, j))
%             s = dot(A(1:i, i), A(1:i, j));
%             d = d + abs(A(i, j) - s);
%         end
%     end
% end
% 

D = spdiags(1 ./ sqrt(diag(A0)), 0, n, n);
A = D * A0 * D;
% 
% alpha = max(sum(abs(A),2)./diag(A))-2;
% chol(A);
% ic = ichol(sparse(A), struct('type','ict','droptol',1e-3,'diagcomp',alpha));
% icf = full(ic)';
% 
[L, U, level] = iluk(D * A0 * D, 5);
[L1, U1, xx] = iluk(D * A0 * D, 1);
% [L10, U10, xx] = iluk(D * A0 * D, 10);

% Compute the difference between the two sparse matrices L and L10
% L1_diff = L1 - L;
% L10_diff = L10 - L;

% Plot the sparsity patterns of L1_diff and L10_diff in the same figure with different markers
% figure;
% hold on;
% spy(L1_diff, 'ro'); % Red dots for L1_diff
% spy(L10_diff, 'b*'); % Blue dots for L10_diff
% title('Sparsity Patterns of L1_diff and L10_diff');
% xlabel('Columns');
% ylabel('Rows');
% legend('L1\_diff', 'L10\_diff');
% hold off;

x_ref = ones(n, 1);
b = A0 * x_ref;
b_scaled = D * b;
% b0 = A * b_scaled;
% r = b_scaled - b0;
% scale = max(r);
% r = r ./ scale;
% y = ic \ r;
% x = icf \ y;
% rho = norm(r);
% r = x ./ rho;
% r = A * r;
% y = ic \ r;
% x = icf \ y;
% x = gmres(sparse(A), D * full(A0) * ((1:n)'), 4, 1e-14, 1, L, U);
gmres(sparse(A), b_scaled, 50, 1e-12, 100, L, U);

nnz(U)

% [ratios, minRatio, avgRatio] = diagonal_dominance_ratio_sparse(A0);
% fprintf("min = %g, avg = %g\n", full(minRatio), full(avgRatio))
% 
% disp(L(1:20, :))

function [ratios, minRatio, avgRatio] = diagonal_dominance_ratio_sparse(A)
    % Ensure A is square
    [n, m] = size(A);
    if n ~= m
        error('Matrix must be square');
    end
    
    % Extract diagonal and absolute values
    diag_vals = abs(diag(A));             % n x 1
    row_sums = sum(abs(A), 2);            % n x 1, sum across each row
    
    % Off-diagonal sums
    off_diag_sums = row_sums - diag_vals;
    
    % Compute ratios (handle division by zero)
    ratios = diag_vals ./ off_diag_sums;
    ratios(off_diag_sums == 0) = Inf;     % Pure diagonal row → infinite ratio
    
    % Summary
    minRatio = min(ratios);
    avgRatio = mean(ratios);
end