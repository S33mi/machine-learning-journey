% =============================================
% Driver for Multiclass Logistic Regression (SGD)
% Updated for Octave / MATLAB
% =============================================

clear; clc;

fprintf('Loading MNIST data for Multiclass...\n');

% Load data
load mnist_uint8.mat

train_x = double(train_x);
train_y = double(train_y);
test_x  = double(test_x);
test_y  = double(test_y);

% Convert one-hot to label format (0 to 9)
[~, train_labels] = max(train_y, [], 2);
train_labels = train_labels - 1;   % 0-based

[~, test_labels] = max(test_y, [], 2);
test_labels = test_labels - 1;

fprintf('Training samples: %d | Test samples: %d | Classes: 10\n', ...
        size(train_x,1), size(test_x,1));

% ================== Train using SGD ==================
fprintf('\nStarting SGD training for Multiclass...\n');

W = train_sgd(train_x, train_labels, ...
              'learning_rate', 0.15, ...
              'epochs', 15, ...
              'batch_size', 128); 
% tweak the learning_rate & epochs for higher accuracy

% Train without batching
%W = train_mlr_full_gd(train_x, train_labels, epochs=80, learning_rate=0.2);

% ================== Test ==================
fprintf('\nEvaluating on test set...\n');
acc = test_sgd(W, test_x, test_labels);

% Save result
save acc_multiclass acc
fprintf('\nDone! Accuracy saved.\n');
