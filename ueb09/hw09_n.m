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
FPH        = [];                            % forward pass history (memory)
EH         = [];                            % error history (memory)
LTD        = [1 1 0; 1 0 1; 0 1 1; 0 0 0];  % labeled training data
L          = [0; 1; 1; 0];                  % labels
ATD        = [1 1 1; 1 0 1; 0 1 1; 0 0 1];  % augmented data without labels
iterations = 2;                             % number of iterations
alpha      = 0.01;                          % learning rate

% set initial values
W1 = W1_init;
W2 = W2_init;
L0 = ATD

% start training
for runs = 1:iterations

    % forward pass:
    [L1,L2] = forward_pass(L0,W1,W2)
    FPH = horzcat(FPH,L2);   % recording the learn process

    % error calculation
    e = error_calculation(L2,L);
    EH = horzcat(EH,e);      % recording the learn process

    % backward pass
    [W1,W2] = backward_pass(alpha,L0,L1,L2,W1,W2,e);

end

FPH
EH



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
