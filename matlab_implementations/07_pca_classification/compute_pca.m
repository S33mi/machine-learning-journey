function [evecs, evals] = compute_pca(X)
    % X is D x N data matrix (D = dimensions, N = samples)
    [D, N] = size(X);
    
    % Step 1: Mean center the data
    mu = mean(X, 2);           % D x 1
    X_centered = X - mu;
    
    % Step 2: Compute covariance matrix (D x D)
    C = (X_centered * X_centered') / (N - 1);
    
    % Step 3: Eigen decomposition
    [V, D_mat] = eig(C);
    evals = diag(D_mat);
    
    % Sort eigenvalues and eigenvectors in descending order
    [evals, idx] = sort(evals, 'descend');
    evecs = V(:, idx);
    
    % Optional: Keep only positive eigenvalues
    pos_idx = evals > 1e-10;
    evecs = evecs(:, pos_idx);
    evals = evals(pos_idx);
end
