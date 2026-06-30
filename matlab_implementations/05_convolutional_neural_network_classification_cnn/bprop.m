function net = bprop(net, activations, y_true, lr)
    delta = activations{5} - y_true';   % gradient of cross-entropy + softmax
    
    % FC layer
    flat = reshape(activations{4}, 192, []);
    net.fc.W = net.fc.W - lr * (flat * delta');
    net.fc.b = net.fc.b - lr * sum(delta, 2)';
    
    % Backprop through pool2, conv2 etc. is more involved (gradients for max-pool + conv)
    % For now, this is a starter. Full version needs delta propagation.
    fprintf('.');  % progress
end
