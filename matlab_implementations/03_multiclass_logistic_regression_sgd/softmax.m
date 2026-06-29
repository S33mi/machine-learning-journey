function p = softmax(X, W)
    scores = X * W;
    scores = scores - max(scores, [], 2);   % stability
    exp_scores = exp(scores);
    p = exp_scores ./ sum(exp_scores, 2);
end
