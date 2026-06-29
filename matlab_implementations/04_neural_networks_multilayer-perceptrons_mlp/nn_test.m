function acc = nn_test(W1, W2, testX, testY)
    Z1 = testX * W1;
    A1 = sigmoid(Z1);
    Z2 = A1 * W2;
    [~, pred] = max(Z2, [], 2);
    pred = pred - 1;   % if labels are 0-based
    
    acc = mean(pred == testY) * 100;
    fprintf('Test Accuracy: %.2f%%\n', acc);
end
