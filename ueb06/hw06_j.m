% Clean up
clear all
close all
clc

% Datenaufbereitung
Data         = load('klausur.txt');
Punkte       = Data(:,1);
Note         = Data(:,2);
Punkte0      = Data((Data(:,2)==0),:);
Punkte1      = Data((Data(:,2)==1),:);


%%%%%%%%%%%%  Aufgabe 1 - Perceptron Learning  %%%%%%%%%%%%
% Suche nach der Geraden w (Normale der Trennlinie zwischen den Klassen 0 und 1):
w = [max(Punkte) max(Note)] % initiales w, per Zufall gesetzt
t = 0;  % Anzahl Iterationen, in denen eine Korrektur vorgenommen wurde
limit = 10;  % max. Anzahl von Iterationen
for i = 1:size(Punkte, 1) % wir laufen einfach alle Punkte durch statt sie zufaellig zu waehlen
    if i > limit
        break
    end
    w_norm = w / norm(w)  % Einheitsvektor zu w berechnen
    proj = Data(i, :) * w_norm' % Skalarprojektion des aktuellen Datenpunktes auf w_norm
    
    if Note(i) == 1  % element aus Klasse 1
        if proj < 0 % element aus Klasse 1 wurde falsch klassifiziert
            t = t + 1
            w = w + Data(i) % Korrektur
        end
    end
    if Note(i) == 0  % element aus Klasse 1
        if proj > 0 % element aus Klasse 1 wurde falsch klassifiziert
            t = t + 1
            w = w - Data(i) % Korrektur
        end
    end
end


%%%%%%%%%  Aufgabe 2 b - lineare Regression  %%%%%%%%%%

% Grafik erstellen
figure('NumberTitle','off','Name','Aufgabe 2 - Lin. Regression');
hold on
scatter(Punkte, Note)

y = Note;  % unsere Labels
onesVector = ones(size(Data,1), 1); % Spaltenvektor mit Einsen der gleichen Laenge wie Data
X = horzcat(onesVector, Punkte);  % Einsen-Vektor an Datenpunkte-Matrix drankleben
beta = inv(X'*X) * X' * y  % beta ist der Vektor, mit dem Eingabedaten multipliziert werden muessen, damit wir an die Klassen-Labels kommen
% beta = [-0.2736; 1.6104]

%  % plot regression line:
%  li = linspace(0, 1);  % create a row vector of 100 evenly spaced points between 0 and 1
%  beta_x = beta(1) * li;
%  beta_y = beta(2) * li;
%  plot(beta_x, beta_y, 'g');

% example: classify a score of 0.4:
0.4 * beta
%  ans =  [-0.1095; 0.6442] - wrong answer