function [activations, net] = fprop(net, X)
    activations = cell(1,6);
    
    [conv1, net.conv1] = conv_layer(X, net.conv1);
    activations{1} = conv1;
    
    [pool1, net.pool1] = pool_layer(conv1, 2);
    activations{2} = pool1;
    
    [conv2, net.conv2] = conv_layer(pool1, net.conv2);
    activations{3} = conv2;
    
    [pool2, net.pool2] = pool_layer(conv2, 2);
    activations{4} = pool2;
    
    flat = reshape(pool2, 192, []);  % 4x4x12
    [probs, net.fc] = fc_layer(flat, net.fc);
    activations{5} = probs;
    
    % activations{6} = probs;  % final output

end
