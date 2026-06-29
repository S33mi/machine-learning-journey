clear; clc;

load mnist_uint8.mat

train_x = double(train_x);
train_y = double(train_y);
test_x = double(test_x);
test_y = double(test_y);

% Use subset for speed
max_samples = 8000; % update to 10000+ if no memory issues
if size(train_x,1) > max_samples
    rng(42);
    idx = randperm(size(train_x,1), max_samples);
    train_x = train_x(idx,:);
    train_y = train_y(idx,:);
end

[~, test_labels] = max(test_y, [], 2);
test_labels = test_labels - 1;

fprintf('Training on %d samples\n', size(train_x,1));

[W1, W2] = nn_train(train_x, train_y, hidden_size=30, epochs=50, alpha=0.15);

% Test Accuracy : 81.67% 
% with max_samples = 8000, hidden_size=30, epochs=50, alpha=0.15
% tweak hidden_size and epochs for higher accuracy

% Test
Z1 = test_x * W1;
A1 = sigmoid(Z1);
Z2 = A1 * W2;
[~, pred] = max(Z2, [], 2);
pred = pred - 1;

acc = mean(pred == test_labels) * 100;
fprintf('Test Accuracy: %.2f%%\n', acc);
