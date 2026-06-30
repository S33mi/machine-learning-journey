function acc = test_cnn(net, test_x, test_labels)
    [~, net] = fprop(net, test_x);
    [~, pred] = max(net.fc.out, [], 1);
    pred = pred';
    true_label = test_labels;  % index from 1-10
    acc = mean(pred == true_label);
end
