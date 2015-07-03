% clean up
clear all
close all
clc

% load data
Training  = load('pendigits-training.txt');
Testing   = load('pendigits-testing.txt');

% center training data
for index1 = 1:length(Training)
   for index2 = 1:16
      cTraining(index1,index2) = Training(index1,index2) / max(Training(index1,1:16));
   end
end

% center test data
for index1 = 1:length(Testing)
   for index2 = 1:16
      cTesting(index1,index2) = Testing(index1,index2) / max(Testing(index1,1:16));
   end
end

%%% Aufgabe 1 - XOR-Netzwerk mit Backpropagation trainieren %%%
W1_init    = rand(3,2);                     % random weights 3x2 from input to layer 1
W2_init    = rand(3,1);                     % random weights 3x1 from layer 1 to layer 2
LTD        = [1 1 0; 1 0 1; 0 1 1; 0 0 0];  % labeled training data
L          = [0; 1; 1; 0];                  % labels
ATD        = [1 1 1; 1 0 1; 0 1 1; 0 0 1];  % augmented data without labels
iterations = 1000;                       % number of iterations
alpha      = 0.01;                          % learning rate

% set initial values
W1                     = W1_init;
W2                     = W2_init;

for rc=1:1000
    % start training
    random                 = randi(4);
    L0                     = ATD(random,:);
    label                  = L(random,:);
    for runs = 1:iterations

        % forward pass
        % layer 1
        t                  = L0 * W1;
        perceptron1_layer1 = 1 / 1 + exp(-t(:,1));
        perceptron2_layer1 = 1 / 1 + exp(-t(:,2));
        out_layer1         = [perceptron1_layer1, perceptron2_layer1];
        
        % layer 2
        t                  = [perceptron1_layer1, perceptron2_layer1, 1]*W2;
        perceptron1_layer2 = 1 / 1 + exp(-t);
        out_layer2         = perceptron1_layer2;
        
        % error calculation
        e                  = perceptron1_layer2 - label;
        
        % backward pass
        t1                 = L0 * W1;
        s11                = (1 / 1+exp(-t1(:,1)))*(1-(1 / 1+exp(-t1(:,1))));
        s12                = (1 / 1+exp(-t1(:,2)))*(1-(1 / 1+exp(-t1(:,2))));
        D1                 = [s11, 0; 0, s12];
        
        t2                 = [out_layer1, 1] * W2;
        s2                 = (1 / 1+exp(-t2))*(1-(1 / 1+exp(-t2)));
        D2                 = s2;
        
        W2_                = W2(1:2,:);
        dW1                = -alpha*D1*W2_*D2*e*L0;
        dW2                = -alpha*D2*e*[out_layer1, 1];
        W1                 = W1 + dW1';
        W2                 = W2 + dW2';

    end

end
quality = 1 - abs(label - e)
label
e

%%% Aufgabe 2 - Handgeschriebene Zahlen klasssifizieren %%%

%  numInput  = 16;  % number of input neurons (without constant factor). These do not compute anything.
%  numHidden =  2   % number of neurons in the hidden layer: 2, 4, 8
%  numOutput = 10;  % number of neurons in the output layer
%  
%  weights1 = rand(numInput + 1, numHidden + 1); % weights of the first layer
%  weights2 = rand(numHidden + 1, numOutput + 1);
%  D1 = eye(3);  % matrix derivatives of the sigmoid function along the diagonal
%  D2 = eye()
%  D1 = zeros(numInput + 1, numHidden + 1);
%  D2 = zeros(numHidden + 1, numHidden + 1);
%  outputs1 = zeros(1, 3);  % outputs of the first layer
%  x = rand(2, 1);  % two randomly chosen input values
%  x = vertcat(x, 1);
%  
%  % Feed Forward step:
