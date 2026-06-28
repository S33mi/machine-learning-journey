function acc = test_lr(w, testX, testt)
    % Add bias
    testX = [ones(size(testX,1),1), testX];
    
    y_prob = sigmoid(testX * w);
    y_pred = y_prob >= 0.5;
    
    acc = mean(y_pred == testt) * 100;
    fprintf('Test Accuracy: %.2f%%\n', acc);
    
    % Confusion Matrix
    C = confusionmat(testt, y_pred);
    disp('Confusion Matrix:');
    disp(C);
end
