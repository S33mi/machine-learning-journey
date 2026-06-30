function y = softmax(x)
    ex = exp(x - max(x,[],1));
    y = ex ./ sum(ex,1);
end
