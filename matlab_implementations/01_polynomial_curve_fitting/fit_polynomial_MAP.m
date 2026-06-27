function w = fit_polynomial_MAP(X, T, M, alpha, beta)
    % (Maximum Posterior)
    % alpha: prior precision on w
    % beta: noise precision (1/sigma^2)
    
    N = length(X);
    Phi = zeros(N, M+1);
    
    for j = 0:M
        Phi(:, j+1) = X .^ j;
    end
    
    % Regularized least squares
    % w = (beta * Phi'Phi + alpha * I)^-1 * (beta * Phi'T)
    lambda = alpha / beta;  % regularization coeff
    I = eye(M+1);
    w = (beta * (Phi'*Phi) + alpha * I) \ (beta * Phi'*T);
end
