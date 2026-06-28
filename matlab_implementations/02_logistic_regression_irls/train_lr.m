function w = train_lr(X, t)
    % X: N x D matrix (no bias yet)
    % t: N x 1 vector of 0/1
    
    [N, D] = size(X);
    
    % Add bias term
    X = [ones(N,1), X];
    
    % IRLS
    w = zeros(D+1, 1);
    max_iter = 30;
    tol = 1e-5;
    
    for iter = 1:max_iter
        y = sigmoid(X * w);
        
        R = diag(y .* (1 - y) + 1e-10);
        z = X * w + (t - y) ./ (y .* (1 - y) + 1e-10);
        
        w_new = (X' * R * X) \ (X' * R * z);
        
        if norm(w_new - w) < tol
            fprintf('IRLS converged in %d iterations.\n', iter);
            break;
        end
        w = w_new;
    end
end
