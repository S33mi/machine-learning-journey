%run this file

clear; clc;

% Load data (adjusted for Octave Online)
load mnist_uint8.mat

train_x = double(train_x);
train_y = double(train_y);
test_x = double(test_x);
test_y = double(test_y);

class1 = 1;
class2 = 2;

% Training data (digits 1 and 2 in 1-based indexing)
class1_inds = find(train_y(:,class1)==1);
class2_inds = find(train_y(:,class2)==1);

n1 = length(class1_inds);
n2 = length(class2_inds);
training_inds = [class1_inds; class2_inds];

X = train_x(training_inds, :);
t = [zeros(n1,1); ones(n2,1)];

% Testing data
class1_inds = find(test_y(:,class1)==1);
class2_inds = find(test_y(:,class2)==1);
testing_inds = [class1_inds; class2_inds];

testX = test_x(testing_inds, :);
testt = [zeros(length(class1_inds),1); ones(length(class2_inds),1)];

fprintf('Training samples: %d\n', length(t));
fprintf('Test samples: %d\n', length(testt));


% Reduce training size for online Octave
max_train = 2000;
if length(t) > max_train
    idx = randperm(length(t), max_train);
    X = X(idx,:);
    t = t(idx);
end

% Train and Test
w = train_lr(X, t);
acc_lr = test_lr(w, testX, testt);

save acc_lr acc_lr
disp('Done! acc_lr saved.');

%*************take aways************
% if you are running online you can get memory issues 
% all code in here will solve this issue
%Reduce max_samples = 800 or even 500



