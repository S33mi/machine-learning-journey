function [out, layer] = conv_layer(X, layer, stride)
    if nargin < 3, stride = 1; end
    [H, W, C, N] = size(X);
    f = size(layer.W,1);  % filter size
    num_f = size(layer.W,4);
    
    out_h = floor((H - f)/stride) + 1;
    out_w = floor((W - f)/stride) + 1;
    out = zeros(out_h, out_w, num_f, N);
    
    for n = 1:N
        for k = 1:num_f
            for c = 1:C
                out(:,:,k,n) = out(:,:,k,n) + conv2(X(:,:,c,n), rot90(layer.W(:,:,c,k),2), 'valid');
            end
            out(:,:,k,n) = out(:,:,k,n) + layer.b(k);
        end
    end
    layer.out = out;  % save for backprop
end
