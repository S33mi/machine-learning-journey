function [out, layer] = fc_layer(X, layer)
    X = reshape(X, [], size(X,4));  % flatten
    out = X' * layer.W + layer.b;
    out = softmax(out')';  % column vectors
    layer.out = out;
end
