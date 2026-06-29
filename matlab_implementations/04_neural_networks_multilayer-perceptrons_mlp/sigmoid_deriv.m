function g = sigmoid_deriv(z)
    g = sigmoid(z) .* (1 - sigmoid(z));
end
