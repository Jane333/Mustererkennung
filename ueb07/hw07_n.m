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
x0100        = linspace(0,100);
N            = length(Data);

%%% Aufgabe 1 - Logistische Regression %%%

alpha = 10^(-7);
beta  = [0,0];   % initiales beta Komponente 2 sollte 1 sein

for repeats = 1:100000
    
    for i = 1:N
    
        k = beta*ExtendedData(i,:)';
        p = exp(k)/(1+exp(k));
    
        % likelihood(beta) = \sum_1^N x_i * ( y_i - p(x_i,beta) )
        likelihood = 0;
        for j = 1:N
            likelihood = likelihood + Distance(j) * ( Goal(j) - p );
        end
    
        beta = beta + alpha * likelihood;
        
        for h = 1:N
            error = 0.5 * (Goal(h) * - p^2);
        end
        
    end
   
    if (repeats / 25000 == 0)
        
        % plot
        figure('NumberTitle','off','Name','Aufgabe 1 - Logistische Regression');
        hold on
        
        gscatter(Data(:,1), ones(size(Punkte,1), 1), Goal,'krb','+x',[],'off');

        title('Aufgabe 1 - Logistische Regression ' + int2str(error));
        xlabel('Distanz zum Tor');
        ylabel('Keine Ahnung')
        axis([-0.1 1.1 -0.1 1.1]);
        legend('Normalenvektor w', 'Diskriminante', 'Nicht bestanden','Bestanden');
        
    end
end