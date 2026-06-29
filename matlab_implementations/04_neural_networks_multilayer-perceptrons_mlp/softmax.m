function p = softmax(z)
    z = z - max(z, [], 2);   % stability
    exp_z = exp(z);
    p = exp_z ./ sum(exp_z, 2);
end
