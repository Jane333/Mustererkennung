% clean up
clear all
close all
clc

% load data
Data   = load('pendigits-training.txt');

% prepare data for network
LData1 = Data(1:60,:);
AData1 = horzcat(Data(1:60,1:16),ones(60,1));
Label1 = Data(1:60,17);

LData2 = Data(61:90,:);
AData2 = horzcat(Data(61:90,1:16),ones(30,1));
Label2 = Data(61:90,17);

%%% NETWORK DEFINITION %%%

% 16 Nodes in Layer0 - Inputlayer
% 16 Nodes in Layer1 - Hiddenlayer
% 10 Nodes in Layer2 - Outputlayer

% weights
W1 = ones(17,16)*(-0.5);
W2 = ones(17,10)*(-0.5);

% learning rate
alpha = 1;


% training
for reruns = 1:1000
    dW1 = zeros(16,17);
    dW2 = zeros(10,17);
    for i = 1:60
        d = AData1(i,:);
        l = Label1(i,:);
        
        % forward pass - layer 1
        t1 = d * W1;
        out_layer1 = 1 ./ (1 + exp(-t1));
        
        % forward pass - layer 2
        t2 = [out_layer1, 1]*W2;
        out_layer2 = 1 ./ (1 + exp(-t2));
        
        % error calculation
        lv = zeros(1,10);
        for j = 1:10
            if l == j
                lv(1,j+1) = 1;
            end
        end
        error = (out_layer2 - lv);
        
        % backward pass - layer 1        
        s1_der = (1 ./ (1+exp(-t1))) .* (1-(1 ./ (1+exp(-t1))));
        D1 = diag(s1_der);

        % backward pass - layer 2
        s2_der = (1 ./ (1+exp(-t2))) .* (1-(1 ./ (1+exp(-t2))));
        D2 = diag(s2_der);
        
        W2_                = W2(1:16,:);
        delta2             = D2*error';
        delta1             = D1*W2_*delta2;
        dW1                = dW1 + -alpha*delta1*d;
        dW2                = dW2 + -alpha*delta2*[out_layer1, 1];
    end
    W1                 = W1 + dW1';
    W2                 = W2 + dW2';
end



% Testing

correctly_predicted = 0;
predictedClass = [];
for runs = 1:length(AData1)
    
    d     = AData1(runs,:)
    l     = Label1(runs)
        
    % forward pass
    % layer 1
    t          = d * W1
    out_layer1 = 1 ./ (1 + exp(-t))
    
    % layer 2
    t          = [out_layer1, 1]*W2
    out_layer2 = 1 ./ (1 + exp(-t))
    
    % prediction calculation
    prediction = 999;  % initial value
    predictionVal = max(out_layer2);
    for index = 1:length(out_layer2)
        if out_layer2(1, index) == predictionVal
            prediction = index - 1
        end
    end
    predictedClass = vertcat(predictedClass, prediction);
    if prediction == l
        correctly_predicted = correctly_predicted + 1;
    end

end

confusionMatrix = zeros(10,10);
for i = 1:length(Label1)
    confusionMatrix(Label1(i)+1, predictedClass(i)+1) = confusionMatrix(Label1(i)+1, predictedClass(i)+1) + 1;
end
confusionMatrix

klass_guete = correctly_predicted / length(Label1)
