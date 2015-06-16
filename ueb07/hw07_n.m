% Clean up
clear all
close all
clc

% Datenaufbereitung
Data        = load('fieldgoal.txt');
ExtendeData = [Data(:,1), ones(size(Data,1), 1)];
Distance    = Data(:,1);
Goal        = Data(:,2);
%Goal0       = Data((Data(:,2)==0),:);
%Goal1       = Data((Data(:,2)==1),:);
x01         = linspace(0,1);
x0100       = linspace(0,100);
N           = length(Data);

%%% Aufgabe 1 - Logistische Regression %%%

alpha = 10^(-7); % Lernkonstante
beta  = [0,0];   % initiales beta?
beta = zeros(1,N+1); % da beta die Dimension R^(N+1)
Gradient = 0;

for i = 1:N
    
    % p(x_i,beta) = beta_0 + beta_1 * x_i
    p = beta(i) + beta(i+1)*Distance(i);
    
    % likelihood(beta) = \sum_1^N x_i * ( y_i - p(x_i,beta) )
    for j = 1:N
        Gradient = Gradient + Distance(j) * ( Goal(j) - p );
    end
    
    % beta_i+1 = beta_i + alpha * likelihood(beta_i)   % likelihood = Gradient
    beta(i+1) = beta(i) + alpha*Gradient;
end

% output
beta
plot(beta, 'r')

% das stimmt noch nicht, aber morgen mehr...


% plotten Sie p(x,beta) fuer Entfernungen zwischen 0 und 100