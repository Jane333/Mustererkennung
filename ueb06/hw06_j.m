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
% Suche nach der Geraden w (Normale der Trennlinie zwischen den Klassen 0 und 1):
w = [max(Punkte) max(Note)] % initiales w, per Zufall gesetzt
t = 0;  % Anzahl Iterationen, in denen eine Korrektur vorgenommen wurde
limit = size(Data, 1);  % max. Anzahl von Iterationen
for i = 1:limit    
    w_norm = w / norm(w);  % Einheitsvektor zu w berechnen
    lineNum = mod(i, size(Features,1))+1;
    proj = Features(lineNum, :) * w_norm'; % Skalarprojektion des aktuellen Datenpunktes auf w_norm
    
    if Note(lineNum) == 1  % element aus Klasse 1
        if proj < 0 % element aus Klasse 1 wurde falsch klassifiziert
            t = t + 1;
            w = w + Features(lineNum) % Korrektur
            w_y1 = w(1) * li;
            w_y2 = w(2) * li;
            w_normale = [-w(2) w(1)]
            w_normale_y = w_normale(2) * li;
            
            figure('NumberTitle','off','Name','Aufgabe 1 - Perceptron Learning');
            plot(w_y1, w_y2, 'g')
%              plot(li, w_y, 'g');
            hold on
            scatter(Punkte, Note)
            hold on
            % w ist die Normale zur Diskriminate. Nun berechnen wir die Diskriminate selbst:
            plot(li, w_normale_y, 'm');
            legend('Normalenvektor','Datenpunkte','Diskriminate');
            xlabel('Erreichte Punkte in Prozent')
            ylabel('Nix')
            title('Aufgabe 1 - Perceptron Learning, pos. Verschiebung')
            axis([-1 2 -1 2]) % Achsenskalierung auf den angegebenen Bereich
        end
    end
    if Note(lineNum) == 0  % element aus Klasse 1
        if proj > 0 % element aus Klasse 1 wurde falsch klassifiziert
            t = t + 1;
            w = w - Features(lineNum) % Korrektur
            w_y = w(2) * li;
            w_normale = [-w(2) w(1)]
            w_normale_y = w_normale(2) * li;
            
            figure('NumberTitle','off','Name','Aufgabe 1 - Perceptron Learning');
            hold on
            plot(li, w_y, 'g');
            hold on
            scatter(Punkte, Note)
            hold on
            % w ist die Normale zur Diskriminate. Nun berechnen wir die Diskriminate selbst:
            plot(li, w_normale_y, 'm');
            legend('Normalenvektor','Datenpunkte','Diskriminate');
            xlabel('Erreichte Punkte in Prozent')
            ylabel('Nix')
            title('Aufgabe 1 - Perceptron Learning, neg. Verschiebung')
            axis([-1 2 -1 2]) % Achsenskalierung auf den angegebenen Bereich
        end
    end
end


%%%%%%%%%  Aufgabe 2 a - Schwellwert fuer Aufg. 1  %%%%%%%%%%
% Bestanden: ab 50 %. Fuer welches x gibt die Geradengleichung fuer den Vektor w ein y = 0.5 zurueck?


%%%%%%%%%  Aufgabe 2 b - lineare Regression  %%%%%%%%%%

% Grafik erstellen
figure('NumberTitle','off','Name','Aufgabe 2 - Lin. Regression');
hold on
scatter(Punkte, Note)

y = Note;  % unsere Labels
onesVector = ones(size(Data,1), 1); % Spaltenvektor mit Einsen der gleichen Laenge wie Data
X = horzcat(onesVector, Punkte);  % Einsen-Vektor an Datenpunkte-Matrix drankleben
beta = inv(X'*X) * X' * y;  % beta ist der Vektor, mit dem Eingabedaten multipliziert werden muessen, damit wir an die Klassen-Labels kommen
% beta = [-0.2736; 1.6104]

%  % plot regression line:
%  beta_x = beta(1) * li;
%  beta_y = beta(2) * li;
%  plot(beta_x, beta_y, 'g');

% example: classify a score of 0.4:
[0.4 0] * beta
% 