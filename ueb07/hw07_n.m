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
limit        = 100000;
plist        = [];

%%% Aufgabe 1 - Logistische Regression %%%

alpha = 10^(-7);
beta  = [0,0];   % initiales beta

for repeats = 1:limit
    
    likelihood = 0;
    e          = 0;
    
    for i = 1:N
    
        k = beta*ExtendedData(i,:)';
        p = exp(k)/(1+exp(k));
        likelihood = likelihood + Distance(i) * ( Goal(i) - p );
        e = e + abs(Goal(i) - p);

    end

    beta = beta + (alpha * likelihood);
    plist = vertcat(plist,p);
    
    if mod(repeats,25000) == 0
        e
    end
end

% Diskriminante
fx = beta(1) * beta(2)*x0100;

% plot
figure('NumberTitle','off','Name','Aufgabe 1 - Logistische Regression');
hold on
  
scatter(Distance, Goal);
plot(plist, 'g');
plot(fx);

title('Aufgabe 1 - Logistische Regression');
xlabel('Distanz zum Tor');
ylabel('Wahrscheinlichkeit für einen Treffer')
axis([-0.1 100.1 -0.1 1.1]);
legend('Datenpunkte','p(x,beta)','Diskriminante');