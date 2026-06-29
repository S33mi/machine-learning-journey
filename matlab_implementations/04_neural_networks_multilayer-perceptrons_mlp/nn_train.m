function [W1, W2] = nn_train(X, Y, hidden_size=100, epochs=100, alpha=0.1)
    % X: N x D
    % Y: N x K (one-hot)
    
    [N, D] = size(X);
    K = size(Y, 2);
    
    % Initialize weights
    W1 = randn(D, hidden_size) * 0.01;   % Input -> Hidden
    W2 = randn(hidden_size, K) * 0.01;   % Hidden -> Output
    
    for epoch = 1:epochs
        % Forward pass
        Z1 = X * W1;
        A1 = sigmoid(Z1);
        Z2 = A1 * W2;
        A2 = softmax(Z2);   % Output probabilities
        
        % Backpropagation
        dZ2 = A2 - Y;       % Error at output
        dW2 = A1' * dZ2 / N;

        %dZ1 = (dZ2 * W2') .* (A1 .* (1 - A1));   % sigmoid derivative
        dZ1 = (dZ2 * W2') .* sigmoid_deriv(Z1);
        dW1 = X' * dZ1 / N;
        
        % Update weights
        W1 = W1 - alpha * dW1;
        W2 = W2 - alpha * dW2;
        
        if mod(epoch, 20) == 0
            fprintf('Epoch %d completed\n', epoch);
        end
    end
end
