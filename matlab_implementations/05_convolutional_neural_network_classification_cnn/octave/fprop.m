function [activations, net] = fprop(net, X)
    activations = cell(1,6);
    
    [conv1_out, net.conv1] = conv_layer(X, net.conv1);
    activations{1} = conv1_out;
    
    [pool1_out, net.pool1] = pool_layer(conv1_out, 2);
    activations{2} = pool1_out;
    
    [conv2_out, net.conv2] = conv_layer(pool1_out, net.conv2);
    activations{3} = conv2_out;
    
    [pool2_out, net.pool2] = pool_layer(conv2_out, 2);
    activations{4} = pool2_out;
    
    % Safe flatten: 4x4x12 = 192
    flat = reshape(pool2_out, 4*4*12, []);
    activations{5} = flat;   % save flattened version too if needed
    
    [probs, net.fc] = fc_layer(flat, net.fc);
    activations{6} = probs;
end
