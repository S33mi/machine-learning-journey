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
    K = 10;                     % 10 classes
    X = [ones(N,1), X];         % bias
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
            
            % Gradient
            grad = zeros(D, K);
            for k = 1:length(Yb)
                label = Yb(k) + 1;
                grad = grad + Xb(k,:)' * (probs(k,:) - (1:K == label));
            end
            grad = grad / length(Yb);
            
            W = W - lr * grad;
        end
        
        if mod(epoch, 5) == 0
            fprintf('Epoch %d completed\n', epoch);
        end
    end
end
