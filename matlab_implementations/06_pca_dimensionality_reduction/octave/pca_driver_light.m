% PCA Assignment - Lightweight version for Octave
clear; clc;

load('mnist_uint8.mat');

% Use small subset to avoid timeout
num_samples = 3000;                    % reduce if still slow
X = double(train_x(1:num_samples, :))'; % D x N = 784 x 3000

% Mean center
mu = mean(X, 2);
X_centered = X - mu;

% Covariance matrix
C = (X_centered * X_centered') / (num_samples - 1);

% Eigen decomposition
[V, D] = eig(C);
evals = diag(D);

% Sort descending
[evals, idx] = sort(evals, 'descend');
evecs = V(:, idx);

fprintf('PCA completed on %d samples\n', num_samples);
fprintf('Top 10 eigenvalues:\n');
disp(evals(1:10)');

% Explained variance
cum_var = cumsum(evals) / sum(evals);
fprintf('Variance explained by top 50 PCs: %.2f%%\n', cum_var(50)*100);

% Save first few eigenvectors for visualization
figure;
for i = 1:9
    subplot(3,3,i);
    img = reshape(evecs(:,i), 28, 28)';
    imagesc(img);
    colormap gray;
    axis image off;
    title(sprintf('PC %d', i));
end
print('pca_eigenvectors.png', '-dpng');

disp('Done! Check pca_eigenvectors.png');
