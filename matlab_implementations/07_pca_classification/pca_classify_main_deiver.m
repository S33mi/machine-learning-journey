% =============================================
% Classification via PCA
% =============================================

clear; clc;

% Load data 
load mnist_uint8.mat

train_x = double(train_x');   % D x N
test_x = double(test_x');

train_y = double(train_y);    % one-hot or labels
test_y = double(test_y);

% Convert one-hot to labels if needed
if size(train_y,2) > 1
    [~, train_labels] = max(train_y, [], 2);
    train_labels = train_labels - 1;
    [~, test_labels] = max(test_y, [], 2);
    test_labels = test_labels - 1;
else
    train_labels = train_y;
    test_labels = test_y;
end

fprintf('Training samples: %d | Test samples: %d\n', size(train_x,2), size(test_x,2));

% ================== Compute PCA (same as 06_pca_dimensionality_reduction) ==================
M = 50;   % Number of principal components (try 20, 50, 100, 200)

[evecs, ~] = compute_pca(train_x);   % Assume this function exists from Assn 5
V = evecs(:, 1:M);                   % Top M eigenvectors

% Mean of training data
mu = mean(train_x, 2);

% ================== Project Training Data ==================
train_centered = train_x - mu;
phi_train = V' * train_centered;     % M x N

% ================== Classify Test Samples (Nearest Neighbour Approach) ==================
correct = 0;
N_test = size(test_x, 2);

for i = 1:N_test
    x = test_x(:, i);
    phi = V' * (x - mu);             % Project test sample
    
    % Euclidean distance to all training projections
    dist = sum((phi_train - phi).^2, 1);
    [~, nn_idx] = min(dist);
    
    predicted_label = train_labels(nn_idx);
    
    if predicted_label == test_labels(i)
        correct = correct + 1;
    end
end

accuracy = correct / N_test * 100;
fprintf('\nFinal Test Accuracy with M=%d: %.2f%%\n', M, accuracy);

% Save result
save accuracy_pca accuracy M
