function net = train_cnn_light(net, X, Y, epochs, batch_size)
    N = size(X,4);
    
    for e = 1:epochs
        idx = randperm(N);
        for b = 1:batch_size:N
            end_b = min(b + batch_size - 1, N);
            batch_x = X(:,:,:, idx(b:end_b));
            batch_y = Y(idx(b:end_b), :);
            
            [acts, net] = fprop(net, batch_x);
            net = bprop(net, acts, batch_y, 0.05);  % higher LR for small data
        end
        fprintf('Epoch %d done\n', e);
    end
end
