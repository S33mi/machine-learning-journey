function net = train_cnn(X, Y, epochs, batch_size)
    net = init_cnn();
    N = size(X,4);
    
    for e = 1:epochs
        idx = randperm(N);
        for b = 1:batch_size:N
            end_b = min(b+batch_size-1, N);
            batch_x = X(:,:,:,idx(b:end_b));
            batch_y = Y(idx(b:end_b), :);
            
            [acts, net] = fprop(net, batch_x);
            net = bprop(net, acts, batch_y, 0.01);
        end
        fprintf('\nEpoch %d completed\n', e);
    end
end
