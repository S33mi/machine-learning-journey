function y = evaluate_polynomial(x, w)
    % x: vector of inputs (Nx1)
    % w: weights ( (M+1) x 1 )
    M = length(w) - 1;
    y = zeros(size(x));
    for j = 0:M
        y = y + w(j+1) * (x .^ j);
    end
end
