% clean up
clear all
close all
clc

% load data
Data   = load('pendigits-training.txt');

% prepare data for network
LData1 = horzcat(Data(1:60,1:16)./100, Data(1:60,17));
AData1 = horzcat(Data(1:60,1:16)./100,ones(60,1));
Label1 = Data(1:60,17);

LData2 = horzcat(Data(61:90,1:16)./100, Data(61:90,17));
AData2 = horzcat(Data(61:90,1:16)./100,ones(30,1));
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

% Aufgabe 1 a) - trainieren bis quadErrorTesting > quadErrorTraining

% training
quadErrorTraining = 0;
quadErrorTesting = 0;
numIter = 0
quadErrorTrainingVector = [];
quadErrorTestingVector = [];
diffVector = [];
%  for numIterations = 1:10000
while quadErrorTraining >= quadErrorTesting
    clc
    [quadErrorTraining,quadErrorTesting, quadErrorTesting - quadErrorTraining]
    numIter = numIter + 1
    
    % start training
    dW1 = zeros(16,17);
    dW2 = zeros(10,17);
    quadErrorTraining = 0;
    quadErrorTesting = 0;
    % start training batch
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
        quadErrorTraining = quadErrorTraining + 0.5*(error * error');
        
        % backward pass - layer 1        
        s1_der = out_layer1 .* (1 - out_layer1);
        D1 = diag(s1_der);

        % backward pass - layer 2
        s2_der = out_layer2 .* (1 - out_layer2);
        D2 = diag(s2_der);
        
        W2_                = W2(1:16,:);
        delta2             = D2*error';
        delta1             = D1*W2_*delta2;
        dW1                = dW1 + -alpha*delta1*d;
        dW2                = dW2 + -alpha*delta2*[out_layer1, 1];
    end
    quadErrorTrainingVector = vertcat(quadErrorTrainingVector, quadErrorTraining);
    W1                 = W1 + dW1';
    W2                 = W2 + dW2';
    
    
    % start testing
    
    for runs = 1:length(AData2)
        
        d     = AData2(runs,:);
        l     = Label2(runs);
            
        % forward pass
        % layer 1
        t1          = d * W1;
        out_layer1 = 1 ./ (1 + exp(-t1));
        
        % layer 2
        t2          = [out_layer1, 1]*W2;
        out_layer2 = 1 ./ (1 + exp(-t2));
        
        % error calculation
        lv = zeros(1,10);
        for j = 1:10
            if l == j
                lv(1,j+1) = 1;
            end
        end
        error = (out_layer2 - lv);
        quadErrorTesting = quadErrorTesting + 0.5*(error * error');
    end
    quadErrorTestingVector = vertcat(quadErrorTestingVector, quadErrorTesting);
    diffVector = vertcat(diffVector, quadErrorTraining - quadErrorTesting);
end % end of while quadErrorTraining >= quadErrorTesting

W1
W2

%  x = linspace(1,10000,10000);
%  figure
%  hold on
%  plot(x, quadErrorTrainingVector);
%  hold on
%  plot(x, quadErrorTestingVector);
%  hold on
%  plot(x, diffVector);
%  xlabel('Anzahl Iterationen');
%  ylabel('Quadratischer Fehler')
%  legend('quadErrorTraining', 'quadErrorTesting', 'diffVector');