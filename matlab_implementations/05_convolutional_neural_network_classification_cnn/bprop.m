function net = bprop(net, activations, y_true, lr)
    % activations{1}=conv1, {2}=pool1, {3}=conv2, {4}=pool2, {5}=fc_probs
    batch_size = size(y_true,1);
    
    % 1. FC Layer
    delta_fc = (activations{5} - y_true') / batch_size;
    flat = reshape(activations{4}, 192, []);  % 4x4x12 = 192
    
    net.fc.W = net.fc.W - lr * (flat * delta_fc');
    net.fc.b = net.fc.b - lr * sum(delta_fc, 2)';
    
    % 2. Back to Pool2
    dpool2 = reshape(delta_fc' * net.fc.W', 4, 4, 12, []);
    dpool2 = pool_backward(dpool2, net.pool2, 2);
    
    % 3. Back to Conv2
    dconv2 = dpool2;  % since no activation in between for simplicity
    net.conv2.W = net.conv2.W - lr * conv_backward(dconv2, activations{2}, [5 5]);
    net.conv2.b = net.conv2.b - lr * sum(sum(sum(dconv2,1),2),4) / batch_size;
    
    % 4. Back to Pool1
    dpool1 = pool_backward(dconv2, net.pool1, 2);  % wait, this is approximate - need proper conv back too
    
    % 5. Back to Conv1 (similar)
    dconv1 = dpool1;
    net.conv1.W = net.conv1.W - lr * conv_backward(dconv1, activations{1}, [5 5]);  % fix indexing
    net.conv1.b = net.conv1.b - lr * squeeze(sum(sum(sum(dconv1,1),2),4))' / batch_size;
end
