% PCA Assignment - Regenerate Bishop Figures 12.3, 12.4, 12.5
clear; clc; close all;

load('mnist_uint8.mat');

% Prepare data: D x N format
X = double(train_x');           % 784 x 60000
X = X(:, 1:10000);              % use 10k samples for speed

% Compute PCA
[evecs, evals] = compute_pca(X);

% 1. Figure 12.3: First few principal components (eigenvectors)
figure('Name', 'Figure 12.3 - Principal Components');
for i = 1:16
    subplot(4,4,i);
    img = reshape(evecs(:,i), 28, 28)';
    imagesc(img);
    colormap gray;
    axis image off;
    title(sprintf('PC %d', i));
end
print('fig_12_3', '-dpng');

% 2. Figure 12.4: Explained variance
figure('Name', 'Figure 12.4 - Explained Variance');
cum_var = cumsum(evals) / sum(evals);
plot(cum_var(1:100), 'LineWidth', 2);
xlabel('Number of principal components');
ylabel('Fraction of variance explained');
grid on;
title('Cumulative Explained Variance');
print('fig_12_4', '-dpng');

% 3. Figure 12.5: Data projection onto first 2 PCs
figure('Name', 'Figure 12.5 - 2D Projection');
proj = evecs(:,1:2)' * (X - mean(X,2));
gscatter(proj(1,:), proj(2,:), train_y(1:size(proj,2)));
xlabel('PC1');
ylabel('PC2');
title('MNIST projected onto first two principal components');
print('fig_12_5', '-dpng');

fprintf('PCA Completed!\n');
fprintf('Top 10 eigenvalues: %s\n', num2str(evals(1:10)', '%.2e '));
fprintf('Variance explained by top 50 PCs: %.2f%%\n', sum(evals(1:50))/sum(evals)*100);
