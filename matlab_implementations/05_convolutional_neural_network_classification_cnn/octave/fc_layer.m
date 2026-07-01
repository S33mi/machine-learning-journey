function [out, layer] = fc_layer(X, layer)
    % X should be 192 x batch_size (flattened pool2)
    if size(X,1) ~= 192
        X = reshape(X, 192, []);   % force correct size 4*4*12
    end
    
    % X' is batch x 192, W is 192 x 10
    scores = X' * layer.W + layer.b;   % batch x 10
    out = softmax(scores')';           % batch x 10
    layer.out = out;
end

function y = softmax(x)
    ex = exp(x - max(x, [], 1));
    y = ex ./ sum(ex, 1);
end
