% Clean up
clear all
close all
clc

% Datenaufbereitung
Data         = load('klausur.txt');
Punkte       = Data(:,1);
Features     = horzcat(Punkte, ones(size(Punkte,1), 1)); % y-Wert jedes Punktes auf 1 setzen, damit Trennlinie durch Ursprung gehen kann
Note         = Data(:,2);
Punkte0      = Data((Data(:,2)==0),:);
Punkte1      = Data((Data(:,2)==1),:);
li = linspace(0, 1);  % create a row vector of 100 evenly spaced points between 0 and 1


%%%%%%%%%%%%  Aufgabe 1 - Perceptron Learning  %%%%%%%%%%%%

disp('Aufgabe 1 - Perceptron Learning')
% Suche nach der Geraden w (Normale der Trennlinie zwischen den Klassen 0 und 1):
w = [0 0] % initiales w, per Zufall gesetzt
limit = size(Data, 1);  % max. Anzahl von Iterationen
for i = 1:limit
    if w(1) == 0 && w(2) == 0
        w_norm = [0 0];
    else
        w_norm = w / norm(w);  % Einheitsvektor zu w berechnen. Wir nehmen alle Korrekturen nur am Einheitsvektor vor, da der Vektor w sonst immer kuerzer/laenger wird durch die Korrekturen
    end
    lineNum = mod(i, size(Features,1))+1;
    proj = Features(lineNum, :) * w_norm'; % Skalarprojektion des aktuellen Datenpunktes auf w_norm
    
    if Note(lineNum) == 1  % element aus Klasse 1
        if proj < 0 % element aus Klasse 1 wurde falsch klassifiziert
            %disp('pos. Verschiebung');
            Features(lineNum, :);
            w = w + Features(lineNum, :); % Korrektur
            w_norm = w / norm(w);  % Einheitsvektor zu w berechnen   1 0            
            w_x = w(1) * li;
            
            % Sonderfallbehandlung fuer w = [1 0]:
            if w(2) == 0
                w_y = w_x * 0;
            else
                coeff_w = w(2) / w(1)  % y = mx  => m = coeff = y/x
                w_y = w_x * coeff_w;
            end
            
            diskriminante = [(-1)*w(2) w(1)];
            diskriminante_x = diskriminante(1) * li;
            if diskriminante(1) == 0  % 0 1
                diskriminante_y = linspace(0, diskriminante(2));
            else
                coeff_d = diskriminante(2) / diskriminante(1);
                diskriminante_y = diskriminante_x * coeff_d;
            end
            
            % Plotten:
            figure('NumberTitle','off','Name','Aufgabe 1 - Perceptron Learning');
            plot(w_x, w_y, 'g');
%              plot(li, w_y, 'g');
            hold on
            gscatter(Punkte, ones(size(Punkte,1), 1), Note,'krb','+x',[],'off');
            hold on
            % w ist die Normale zur Diskriminate. Nun berechnen wir die Diskriminante selbst:
            plot(diskriminante_x, diskriminante_y, 'm');
            legend('Normalenvektor w','Nicht bestanden','Bestanden','Diskriminante');
            xlabel('Erreichte Punkte in Prozent');
            ylabel('Nix');
            title('Aufgabe 1 - Perceptron Learning, pos. Verschiebung');
            axis([-1 2 -1 2]); % Achsenskalierung auf den angegebenen Bereich
            xL = xlim; % get x axis limit
            yL = ylim; % get y axis limit
            hold on
            plot([0 0], yL, ':');  % draw x-axis
            hold on
            plot(xL, [0 0], ':');  % draw y-axis
        end
    end
    if Note(lineNum) == 0  % element aus Klasse 1
        if proj >= 0 % element aus Klasse 1 wurde falsch klassifiziert
            %disp('NEG. Verschiebung');
            w = w - Features(lineNum, :); % Korrektur
            w_norm = w / norm(w);  % Einheitsvektor zu w berechnen
            coeff_w = w(2) / w(1);  % y = mx  => m = coeff = y/x
            w_x = w(1) * li;
            w_y = w_x * coeff_w;
            diskriminante = [(-1)*w(2) w(1)];
            coeff_d = diskriminante(2) / diskriminante(1);
            diskriminante_x = diskriminante(1) * li;
            diskriminante_y = diskriminante_x * coeff_d;
            
            % Plotten:
            figure('NumberTitle','off','Name','Aufgabe 1 - Perceptron Learning');
            hold on
            plot(w_x, w_y, 'g');
            hold on
            gscatter(Punkte, ones(size(Punkte,1), 1), Note,'krb','+x',[],'off');
            hold on
            % w ist die Normale zur Diskriminate. Nun berechnen wir die Diskriminante selbst:
            plot(diskriminante_x, diskriminante_y, 'm');
            legend('Normalenvektor w','Nicht bestanden','Bestanden','Diskriminante');
            xlabel('Erreichte Punkte in Prozent');
            ylabel('Nix');
            title('Aufgabe 1 - Perceptron Learning, neg. Verschiebung');
            axis([-1 2 -1 2]); % Achsenskalierung auf den angegebenen Bereich
            xL = xlim; % get x axis limit
            yL = ylim; % get y axis limit
            hold on
            plot([0 0], yL, ':');  % draw x-axis
            hold on
            plot(xL, [0 0], ':');  % draw y-axis
        end
    end
end


%%%%%%%%%  Aufgabe 2 a - Schwellwert fuer Aufg. 1  %%%%%%%%%%
% Bestanden: ab 50 %. Fuer welches x gibt die Geradengleichung fuer den Vektor w ein y = 0.5 zurueck?

disp('Aufgabe 2 a - Schwellwerte fuer Aufg. 1')
schwellwerte = [];
for iter = 1:100
    randOrder = randperm(size(Features, 1)); % get a line vector consisting of a random permutation of all numbers between 1 and size(Features,1). Ohne zuruecklegen.
    randFeatures = Features(randOrder');
    % Suche nach der Geraden w (Normale der Trennlinie zwischen den Klassen 0 und 1):
    w = [max(Punkte) max(Note)]; % initiales w, per Zufall gesetzt
    t = 0;  % Anzahl Iterationen, in denen eine Korrektur vorgenommen wurde
    limit = size(Data, 1);  % max. Anzahl von Iterationen
    for i = 1:limit
        w_norm = w / norm(w);  % Einheitsvektor zu w berechnen
        lineNum = mod(i, size(randFeatures,1))+1;
        proj = randFeatures(lineNum, :) * w_norm'; % Skalarprojektion des aktuellen Datenpunktes auf w_norm
        
        if Note(lineNum) == 1  % element aus Klasse 1
            if proj < 0 % element aus Klasse 1 wurde falsch klassifiziert
                t = t + 1;
                w = w + randFeatures(lineNum, :); % Korrektur
                diskriminante = [(-1)*w(2) w(1)];
            end
        end
        if Note(lineNum) == 0  % element aus Klasse 0
            if proj >= 0 % element aus Klasse 1 wurde falsch klassifiziert
                t = t + 1;
                w = w - randFeatures(lineNum, :); % Korrektur
                diskriminante = [(-1)*w(2) w(1)];
            end
        end
    end
    schwellwerte = vertcat(schwellwerte, w);
end
mean_schwellwert = mean(schwellwerte)  % der durchschnitliche Schwellwert



%%%%%%%%%  Aufgabe 2 b - lineare Regression  %%%%%%%%%%

disp('Aufgabe 2 b - Lineare Regression')

% Grafik erstellen
figure('NumberTitle','off','Name','Aufgabe 2 - Lin. Regression');
hold on
scatter(Punkte, Note)
hold on

y = Note;  % unsere Labels
onesVector = ones(size(Data,1), 1); % Spaltenvektor mit Einsen der gleichen Laenge wie Data
X = horzcat(onesVector, Punkte);  % Einsen-Vektor an Datenpunkte-Matrix drankleben
beta = inv(X'*X) * X' * y;  % beta ist der Vektor, mit dem Eingabedaten multipliziert werden muessen, damit wir an die Klassen-Labels kommen
% beta = [-0.2736; 1.6104]

% plot regression line:
estimatedLine = X * beta;
estimatedData = horzcat(Data(:,1), estimatedLine);
plot(estimatedData(:,1), estimatedData(:,2), 'g');

% example: classify a score of 0.4:
[0.4 0] * beta;
