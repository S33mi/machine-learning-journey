function W = train_sgd(X, Y, varargin)
    p = inputParser;
    addParameter(p, 'learning_rate', 0.1);
    addParameter(p, 'epochs', 10);
    addParameter(p, 'batch_size', 64);
    parse(p, varargin{:});
    
    lr = p.Results.learning_rate;
    epochs = p.Results.epochs;
    batch_size = p.Results.batch_size;
    
    [N, D] = size(X);
    K = 10;
    X = [ones(N,1), X];
    D = D + 1;
    
    W = zeros(D, K);
    
    for epoch = 1:epochs
        idx = randperm(N);
        X = X(idx,:);
        Y = Y(idx);
        
        for i = 1:batch_size:N
            j = min(i + batch_size - 1, N);
            Xb = X(i:j, :);
            Yb = Y(i:j);
            
            % Forward
            probs = softmax(Xb, W);
            
            % Gradient (vectorized)
            onehot = zeros(length(Yb), K);
            for k = 1:length(Yb)
                onehot(k, Yb(k)+1) = 1;
            end
            grad = Xb' * (probs - onehot) / length(Yb);
            
            W = W - lr * grad;
        end
        
        if mod(epoch, 5) == 0
            fprintf('Epoch %d done\n', epoch);
        end
    end
end
