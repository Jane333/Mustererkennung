% Clean up
clear all
close all
clc

% Datenaufbereitung
Data     = load('klausur.txt');
Punkte   = Data(:,1);
Features = horzcat(Punkte, ones(size(Punkte,1), 1));
Noten    = Data(:,2);
Punkte0  = Data((Data(:,2)==0),:);
Punkte1  = Data((Data(:,2)==1),:);
x1       = linspace(0,1);
x2       = linspace(-5,5);

%%%%%%%%%%%%  Aufgabe 1 - Perceptron Learning  %%%%%%%%%%%%

w = [0 0]; % random choosen vector w
limit = size(Data, 1);  % number of iterations | Abbruchkriterium
for i = 1:limit
    if w(1) == 0 && w(2) == 0
        w_norm = [0 0];
    else
        w_norm = w / norm(w);
    end
    lineNum = mod(i, size(Features,1))+1;
    proj = Features(lineNum, :) * w_norm'; % scalar projection
    
    if Noten(lineNum) == 1
        if proj < 0 % wrong classification
            
            Features(lineNum, :);
            w = w + Features(lineNum, :);
            w_norm = w / norm(w);
            w_x = w(1) * x1;
            
            if w(2) == 0
                w_y = w_x * 0;
            else
                coeff_w = w(2) / w(1)
                w_y = w_x * coeff_w;
            end
            
            diskriminante = [-w(2) w(1)];
            diskriminante_x = diskriminante(1) * x1;
            
            if diskriminante(1) == 0
                diskriminante_y = linspace(0, diskriminante(2));
            else
                coeff_d = diskriminante(2) / diskriminante(1);
                diskriminante_y = diskriminante_x * coeff_d;
            end
            
            % plot
            figure('NumberTitle','off','Name','Aufgabe 1 - Perceptron Learning');
            hold on            
            
            plot(w_x, w_y, 'g');
            plot(diskriminante_x, diskriminante_y, 'm');
            gscatter(Punkte, ones(size(Punkte,1), 1), Noten,'krb','+x',[],'off');
                        
            title('Aufgabe 1 - Perceptron Learning, pos. Verschiebung');
            xlabel('Erreichte Punkte in Prozent');
            axis([-0.1 1.1 -0.1 1.1]);
            legend('Normalenvektor w', 'Diskriminante', 'Nicht bestanden','Bestanden');
            
            xL = xlim;
            yL = ylim;
            plot([0 0], yL, ':');
            plot(xL, [0 0], ':');
        end
    end
    if Noten(lineNum) == 0
        if proj >= 0 % wrong classification
            
            w = w - Features(lineNum,:);
            w_norm = w / norm(w);
            coeff_w = w(2) / w(1);
            w_x = w(1) * x1;
            w_y = w_x * coeff_w;
            diskriminante = [-w(2) w(1)];
            coeff_d = diskriminante(2) / diskriminante(1);
            diskriminante_x = diskriminante(1) * x1;
            diskriminante_y = diskriminante_x * coeff_d;
            
            % plot
            figure('NumberTitle','off','Name','Aufgabe 1 - Perceptron Learning');
            hold on
            
            plot(w_x, w_y, 'g');
            plot(diskriminante_x, diskriminante_y, 'm');
            gscatter(Punkte, ones(size(Punkte,1), 1), Noten,'krb','+x',[],'off');
                   
            title('Aufgabe 1 - Perceptron Learning, neg. Verschiebung');
            xlabel('Erreichte Punkte in Prozent');
            axis([-0.1 1.1 -0.1 1.1]);
            legend('Normalenvektor w', 'Diskriminante', 'Nicht bestanden', 'Bestanden');
            
            xL = xlim;
            yL = ylim;
            plot([0 0], yL, ':');
            plot(xL, [0 0], ':');
        end
    end
end


%%%%%%%%%  Aufgabe 2a - Schwellwert fuer Aufgabe 1  %%%%%%%%%%

schwellwerte = [];
for iter = 1:100
    randOrder = randperm(size(Features, 1));
    randFeatures = Features(randOrder');
    w = [max(Punkte) max(Noten)]; % random choosen vector w
    t = 0;
    limit = size(Data, 1); % number of iterations
    for i = 1:limit
        w_norm = w / norm(w);
        lineNum = mod(i, size(randFeatures,1))+1;
        proj = randFeatures(lineNum, :) * w_norm'; % scalar projection
        if Noten(lineNum) == 1
            if proj < 0
                t = t + 1;
                w = w + randFeatures(lineNum, :);
                diskriminante = [(-1)*w(2) w(1)];
            end
        end
        if Noten(lineNum) == 0
            if proj >= 0 % wrong classification
                t = t + 1;
                w = w - randFeatures(lineNum, :);
                diskriminante = [(-1)*w(2) w(1)];
            end
        end
    end
    schwellwerte = vertcat(schwellwerte, w);
end
schwellwerte
mean_schwellwert = mean(schwellwerte)
% mean_schwellwert = [-0.1980 -0.1880]


%%%%%%%%%  Aufgabe 2b - Lineare Regression  %%%%%%%%%%

figure('NumberTitle','off','Name','Aufgabe 2 - Lin. Regression');

% calculate
onesVector = ones(size(Data,1), 1);
X = horzcat(onesVector, Punkte);
beta = inv(X'*X) * X' * Noten;
fx = beta(1) + beta(2)*x2;
pkt = (0.5-beta(1))/beta(2)
% pkt = 0.4804

% plot
hold on
scatter(Punkte, Noten, 'x', 'b')
plot (x2,fx, 'g')
scatter(pkt,0.5, 'o', 'r')

axis([-0.1 1.1 -0.1 1.1]);
legend('Noten', 'Diskriminante', 'Schwellenwert');
            
