% cnn_driver.m
clear; clc; close all;

load('mnist_uint8.mat');

% Preprocess
train_images = double(reshape(train_x', 28, 28, 1, [])) / 255;   % 60000 x 28 x 28 x 1
test_images  = double(reshape(test_x', 28, 28, 1, [])) / 255;

% One-hot labels
train_labels = zeros(size(train_y,1), 10);
test_labels  = zeros(size(test_y,1), 10);
for i = 1:size(train_y,1)
    train_labels(i, train_y(i)+1) = 1;
end
for i = 1:size(test_y,1)
    test_labels(i, test_y(i)+1) = 1;
end

fprintf('Data loaded. Training CNN...\n');

% Train
net = train_cnn(train_images, train_labels, 5, 64);  % epochs, batch_size

% Test
acc = test_cnn(net, test_images, test_y);
fprintf('Final Test Accuracy: %.2f%%\n', acc*100);

save('cnn_trained.mat', 'net');
