function dpool = pool_backward(dout, pool_layer, pool_size)
    [H, W, C, N] = size(dout);
    dpool = zeros(H*pool_size, W*pool_size, C, N);
    
    for n = 1:N
        for c = 1:C
            for i = 1:H
                for j = 1:W
                    idx = pool_layer.max_idx{i,j,c,n};
                    row = (i-1)*pool_size + idx(1);
                    col = (j-1)*pool_size + idx(2);
                    dpool(row, col, c, n) = dout(i,j,c,n);
                end
            end
        end
    end
end
