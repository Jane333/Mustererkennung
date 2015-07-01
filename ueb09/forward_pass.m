function [layer1,layer2] = forward_pass(L0,W1,W2)

    % i: input values
    % W1: weight matrix for layer 1
    % W2: weight matrix for layer 2
    
    % layer 1
    out_layer_1 = [];
    for index = 1:length(L0)
        out_layer_1 = vertcat(out_layer_1, [perceptron(L0(index,:), W1(:,1)) perceptron(L0(index,:), W1(:,2))]);
    end
    layer1 = out_layer_1;
    out_layer_1 = horzcat(out_layer_1, ones(length(out_layer_1),1));
    
    out_layer_2 = [];
    for index = 1:length(out_layer_1)
        out_layer_2 = vertcat(out_layer_2, [perceptron(L0(index,:), W2(:,1))]);
    end
    layer2 = out_layer_2;

end