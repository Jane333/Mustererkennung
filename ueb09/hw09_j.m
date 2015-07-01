% Clean up
clear all
close all
clc

Training = load('pendigits-training.txt');
Testing = load('pendigits-testing.txt');

%%% Aufgabe 1 - XOR-Netzwerk mit Backpropagation trainieren %%%
W = rand(3,3);
Training = [1 1 0; 1 0 1; 0 1 1; 0 0 0];
ATD = [1 1 1; 1 0 1; 0 1 1; 0 0 1];  % augmentierte Trainingsdaten
numIterations = 1;

% Forward Pass:
error = 0;
for i=1:numIterations
    out1 = [];
    for j = 1:4
        out1 = vertcat(out1, [perceptron(ATD(j,:), W(:,1)) perceptron(ATD(j,:), W(:,2))]);
    end
    out1
    out1aug = horzcat(out1, [1; 1; 1; 1])
    out2 = [];
    for k = 1:4
        out2 = vertcat(out2, perceptron(out1aug(k,:), W(:,3)));
    end
    out2
    error = [out2(1); out2(2) - 1; out2(3) - 1; out2(4)]
end

% Backpropagation:


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
