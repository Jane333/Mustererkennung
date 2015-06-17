% Clean up
clear all
close all
clc

% Datenaufbereitung
Data         = load('fieldgoal.txt');
ExtendedData = [Data(:,1), ones(size(Data,1), 1)];
Distance     = Data(:,1);
Goal         = Data(:,2);
x01          = linspace(0,1);
xminmax      = min(Data(:,1)):max(Data(:,1));
x0100        = linspace(0,100);
N            = length(Data);

%%% Aufgabe 1 - Logistische Regression %%%

alpha = 10^(-7);
beta  = [0,0];   % initiales beta Komponente 2 sollte 1 sein

for repeats = 1:10
    likelihood = 0;
    error = 0;
    for i = 1:N
        k = beta*ExtendedData(i,:)';
        p = exp(k)/(1+exp(k));

        likelihood = likelihood + Distance(i) * ( Goal(i) - p );
        error = error + abs(Goal(i) - p);
    end
    beta = beta + alpha * likelihood;    
   
    if (repeats / 25000 == 0) || (repeats / 50000 == 0) || (repeats / 100000 == 0) || repeats == 10 || repeats == 5
        disp('GesamtError *************************************************************:')
        error
        % plot
        figure('NumberTitle','off','Name','Aufgabe 1 - Logistische Regression');
        scatter(Data(:,1), Data(:,2));
        hold on
        
        
        
        estimatedLine = ExtendedData * beta';
        estimatedData = horzcat(Data(:,1), estimatedLine);
        plot(estimatedData(:,1), estimatedData(:,2), 'g');
        
%          Jana:
%          beta_x = Data(:,1);
%          beta_y = beta_x * beta;
        
%          Nico:
%          fx = beta(1) + beta(2)*x0100;
%          plot(exp(fx));
        
        
        
        
        title('Aufgabe 1 - Logistische Regression ');
        xlabel('Distanz zum Tor');
        ylabel('Keine Ahnung')
    end
end