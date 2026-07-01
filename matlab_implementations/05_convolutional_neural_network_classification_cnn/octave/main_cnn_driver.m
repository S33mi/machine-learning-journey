% cnn_driver_light.m - Lightweight version for Octave Online
clear; clc;

load('mnist_uint8.mat');

% === Use only small subset to avoid memory/time issues ===
num_train = 2000;   % reduce this if still slow (e.g. 1000)
num_test  = 500;

% Training data
train_images = double(reshape(train_x(1:num_train,:)', 28, 28, 1, [])) / 255;
train_y_small = train_y(1:num_train);

% Test data
test_images = double(reshape(test_x(1:num_test,:)', 28, 28, 1, [])) / 255;
test_y_small = test_y(1:num_test);

% One-hot labels
train_labels = zeros(num_train, 10);
for i = 1:num_train
    train_labels(i, train_y_small(i)+1) = 1;
end

fprintf('Using %d training and %d test samples...\n', num_train, num_test);

% Initialize and Train (very few epochs)
net = init_cnn();

epochs = 3;          % keep low to 2,1
batch_size = 32;     % small batch

net = train_cnn_light(net, train_images, train_labels, epochs, batch_size);

% Test
acc = test_cnn(net, test_images, test_y_small);
fprintf('\nTest Accuracy: %.2f%%\n', acc*100);

save('cnn_light.mat', 'net');
