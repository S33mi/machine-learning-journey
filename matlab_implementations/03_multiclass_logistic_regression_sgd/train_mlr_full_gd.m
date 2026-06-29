function W = train_mlr_full_gd(X, Y, epochs=100, learning_rate=0.15)
    % X: N x D (features)
    % Y: N x 1 (labels 0-9)
    
    [N, D] = size(X);
    K = 10;                       % 10 classes
    X = [ones(N,1), X];           % Add bias term
    D = D + 1;
    
    W = zeros(D, K);              % Weight matrix
    
    for epoch = 1:epochs
        % Forward pass
        scores = X * W;
        probs = softmax(X, W);
        
        % One-hot encoding
        onehot = zeros(N, K);
        for i = 1:N
            onehot(i, Y(i)+1) = 1;
        end
        
        % Gradient
        grad = (X' * (probs - onehot)) / N;
        
        % Update weights
        W = W - learning_rate * grad;
        
        if mod(epoch, 10) == 0
            fprintf('Epoch %d completed\n', epoch);
        end
    end
end
