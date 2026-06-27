function w = fit_polynomial_ML(X, T, M)
    % (Maximum Likelihood)
    % X: input vector (N x 1)
    % T: target vector (N x 1)
    % M: polynomial degree
    
    N = length(X);
    Phi = zeros(N, M+1);  % Design matrix
    
    for j = 0:M
        Phi(:, j+1) = X .^ j;
    end
    
    % ML solution: w = (Phi' * Phi)^-1 * Phi' * T
    w = Phi \ T;   % More stable than inv(Phi'*Phi)*Phi'*T
end
