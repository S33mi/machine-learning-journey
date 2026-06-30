function net = init_cnn()
    net.conv1 = struct('W', randn(5,5,1,6)*sqrt(2/(5*5*1)), 'b', zeros(1,6));
    net.conv2 = struct('W', randn(5,5,6,12)*sqrt(2/(5*5*6)), 'b', zeros(1,12));
    net.fc    = struct('W', randn(192,10)*sqrt(2/192), 'b', zeros(1,10));  % 4*4*12 = 192
end
