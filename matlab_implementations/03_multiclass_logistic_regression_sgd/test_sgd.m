function acc = test_sgd(W, testX, testY)
    testX = [ones(size(testX,1),1), testX];
    probs = softmax(testX, W);
    [~, pred] = max(probs, [], 2);
    pred = pred - 1;   % 0-based
    
    acc = mean(pred == testY) * 100;
    fprintf('Test Accuracy: %.2f%%\n', acc);
    
    C = confusionmat(testY, pred);
    disp('Confusion Matrix (diagonal shows correct):');
    disp(diag(C)');
end
