function dW = conv_backward(dout, X, W_size)
    [H, W, C, N] = size(X);
    f = W_size(1);
    num_f = size(dout,3);
    dW = zeros(f, f, C, num_f);
    
    for n = 1:N
        for k = 1:num_f
            for c = 1:C
                dW(:,:,c,k) = dW(:,:,c,k) + conv2(X(:,:,c,n), rot90(dout(:,:,k,n),2), 'valid');
            end
        end
    end
    dW = dW / N;  % average over batch
end
