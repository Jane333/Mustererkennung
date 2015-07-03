% Clean up
clear all
close all
clc

Training = load('pendigits-training.txt');
Testing = load('pendigits-testing.txt');

%%% Aufgabe 1 - XOR-Netzwerk mit Backpropagation trainieren (online-Verfahren) %%%
W1 = rand(3,2) % all weights in the same column lead to the same neuron, all weights in the same row lead from the same input
W2 = rand(3,1)
Training0 = [1 1 0; 1 0 1; 0 1 1; 0 0 0];
ATD0 = [1 1 1; 1 0 1; 0 1 1; 0 0 1];  % augmentierte Trainingsdaten
Training = [];
ATD = [];
for i=1:10000
    random                 = randi(4);
    Training               = vertcat(Training, Training0(random,:));
    random                 = randi(4);
    ATD               = vertcat(ATD, ATD0(random,:));
end

error = 0;
g = 0.1;  % learning rate
D1 = eye(2); % derivatives of the sigmoid function of the first layer
D2 = eye(1);
for i=1:size(Training, 1)
    % Forward Pass:
    in0 = W1' * ATD(i,:)';  % 2x1 = 2x3 * 3x1
    out1 = fw_perceptron_j(in0); % 2x1 - pass the input parameters to the perceptrons of the first layer
    out1 = vertcat(out1, 1);  % 3x1
    D1 = bw_perceptron_j(D1, in0);
    
    in1 = W2' * out1;  % 1x1 = 1x3 * 3x1
    out2 = fw_perceptron_j(in1);    % 1x1
    D2 = bw_perceptron_j(D2, in1);  % 1x1
    
    % Backpropagation:
    error = 0.5 * (out2 - Training(i,3))^2;
    delta2 = D2 * error;           % 1x1
    deltaW2 = -g * delta2 * out1;  % 3x1
    W2 = W2 - deltaW2;             % 3x1
    
    W2_ = W2(1:2,:);
    delta1 = D1 * W2_ * delta2;    % 3x1
    deltaW1 = -g * delta1 * ATD(i,:); % 3x1
    W1 = W1 - deltaW1';  % dies ist falsch, die x'e  muessen von der Spalte abgezogen werden, nicht von der Zeile
end
W1
W2
error

% some tests:
test = [1 1 0; 1 0 1; 0 1 1; 0 0 0];
testaug = [1 1 1; 1 0 1; 0 1 1; 0 0 1];
for i=1:4
    % Forward Pass:
    in0 = W1' * testaug(i)';
    out1 = fw_perceptron_j(in0); % pass the input parameters to the perceptrons of the first layer
    out1 = vertcat(out1, 1);
    
    in1 = W2' * out1;
    out2 = fw_perceptron_j(in1)
end