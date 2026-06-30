function [out, layer] = pool_layer(X, pool_size)
    [H, W, C, N] = size(X);
    out_h = floor(H / pool_size);
    out_w = floor(W / pool_size);
    out = zeros(out_h, out_w, C, N);
    layer.max_idx = cell(out_h, out_w, C, N);  % for backprop
    
    for n = 1:N
        for c = 1:C
            for i = 1:out_h
                for j = 1:out_w
                    h_start = (i-1)*pool_size + 1;
                    w_start = (j-1)*pool_size + 1;
                    patch = X(h_start:h_start+pool_size-1, w_start:w_start+pool_size-1, c, n);
                    [max_val, idx] = max(patch(:));
                    out(i,j,c,n) = max_val;
                    [row, col] = ind2sub([pool_size, pool_size], idx);
                    layer.max_idx{i,j,c,n} = [row, col];
                end
            end
        end
    end
end
