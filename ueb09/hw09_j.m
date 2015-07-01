% Clean up
clear all
close all
clc

%%% Aufgabe 1 - XOR-Netzwerk mit Backpropagation trainieren %%%

numInput  = 16;  % number of input neurons (without constant factor). These do not compute anything.
numHidden =  2   % number of neurons in the hidden layer: 2, 4, 8
numOutput = 10;  % number of neurons in the output layer

weights1 = rand(numInput + 1, numHidden + 1); % weights of the first layer
weights2 = rand(numHidden + 1, numOutput + 1);
D1 = zeros(numInput + 1, numHidden + 1);  % matrix derivatives of the sigmoid function along the diagonal
D2 = zeros(numHidden + 1, numHidden + 1);

% Feed Forward step:
x = rand(2, 1);  % two randomly chosen input values
