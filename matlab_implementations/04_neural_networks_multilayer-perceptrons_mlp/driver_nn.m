clear; clc;

load mnist_uint8.mat

train_x = double(train_x);
train_y = double(train_y);
test_x  = double(test_x);
test_y  = double(test_y);

% Convert to one-hot for training
K = 10;
Y_train = zeros(size(train_y,1), K);
for i = 1:size(train_y,1)
    Y_train(i, find(train_y(i,:)==1)) = 1;
end

% Train
[W1, W2] = nn_train(train_x, Y_train, hidden_size=80, epochs=200, alpha=0.15);

% Test
acc = nn_test(W1, W2, test_x, test_y);   % adjust testY if needed
