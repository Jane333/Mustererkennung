% Clean up
clear all
close all
clc

% Datenaufbereitung
Data         = load('fieldgoal.txt');
ExtendedData = [Data(:,1), ones(size(Data,1), 1)];
Distance     = Data(:,1);
Goal         = Data(:,2);
N            = length(Data);
limit        = 100000;
x_range        = linspace(0,100);
x_print        = [0:100];

%%% Aufgabe 1 - Logistische Regression %%%

figure('NumberTitle','off','Name','Aufgabe 1 - Logistische Regression');

alpha = 10e-7;
beta  = [0;0];   % initiales beta

for i = 1:limit
    
    t = beta' * ExtendedData';
    p = exp(t')./(1.+exp(t'));
    
    likelihood = ExtendedData' * ( Goal - p );

    beta = beta + (alpha * likelihood);
    
    if mod(i,25000) == 0
        
        % Fehler berechnen
        e = sum(abs(Goal - p))
        
        % Wahrscheinlichkeit für einen Treffer berechnen
        p_estimate = 1./(1.+exp(-(beta(2)+ beta(1)*x_print)));
        
        % plot
        hold off
        scatter(Distance, Goal);
        hold on
        plot(x_print,p_estimate,'r');
        
        title('Aufgabe 1 - Logistische Regression');
        xlabel('Distanz zum Tor');
        ylabel('Wahrscheinlichkeit für einen Treffer')
        axis([-0.1 100.1 -0.1 1.1]);
        legend('Datenpunkte','p(x,beta)');
        
        pause(0.0001) % gib Matlab Zeit zu plotten!
    end
end