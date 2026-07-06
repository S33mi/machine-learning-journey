% =============================================
% PCA Classification - Small Subset for Octave
% =============================================

clear; clc;

fprintf('Loading MNIST...\n');
load mnist_uint8.mat

train_x = double(train_x');   % D x N
test_x = double(test_x');

% Convert to labels
[~, train_labels] = max(train_y, [], 2);
train_labels = train_labels - 1;

[~, test_labels] = max(test_y, [], 2);
test_labels = test_labels - 1;

% ================== Use Small Subset ==================
max_train = 2000;   % Small for memory
max_test = 500;

rng(42);
train_idx = randperm(size(train_x,2), max_train);
test_idx = randperm(size(test_x,2), max_test);

train_x = train_x(:, train_idx);
train_labels = train_labels(train_idx);

test_x = test_x(:, test_idx);
test_labels = test_labels(test_idx);

fprintf('Using %d train and %d test samples\n', max_train, max_test);

% ================== PCA ==================
M = 30;   % Number of principal components

[evecs, ~] = compute_pca(train_x);
V = evecs(:, 1:M);

mu = mean(train_x, 2);

% Project training data
train_centered = train_x - mu;
phi_train = V' * train_centered;

% ================== Classify Test Samples ==================
correct = 0;

for i = 1:length(test_labels)
    x = test_x(:, i);
    phi = V' * (x - mu);
    
    dist = sum((phi_train - phi).^2, 1);
    [~, nn_idx] = min(dist);
    
    if train_labels(nn_idx) == test_labels(i)
        correct = correct + 1;
    end
end

accuracy = correct / length(test_labels) * 100;
fprintf('\nPCA Classification Accuracy (M=%d): %.2f%%\n', M, accuracy);
% Using 2000 train and 500 test samples PCA Classification Accuracy (M=30): 93.80%

% Save
save pca_result accuracy M
fprintf('Result saved!\n'); 

