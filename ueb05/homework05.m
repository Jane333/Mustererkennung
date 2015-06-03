% Clean up
clear all
close all
clc

%%%%%%%%%%%%%%%%%  Datenaufbereitung  %%%%%%%%%%%%%%%%

A = load('fisher.txt');
Koordinaten = A(:,1:2);
Klassen = A(:,3);

%%%%%%%%%%%%%%%%%%%%%  Aufgabe 2  %%%%%%%%%%%%%%%%%%%%

%  %  % Create a default (linear) discriminant analysis classifier:
%  %  linclass = fitcdiscr(Koordinaten, Klassen)
%  %  % Classify:
%  %  meanmeas = mean(Koordinaten)
%  %  meanclass = predict(linclass,Koordinaten)

% Datenaufbreitung
Data         = load('fisher.txt');
Data0        = Data((Data(:,3)==0),:);
Data1        = Data((Data(:,3)==1),:);
Koordinaten  = Data(:,1:2);
Koordinaten0 = Data((Data(:,3)==0),1:2);
Koordinaten1 = Data((Data(:,3)==1),1:2);
Klassen      = Data(:,3);

% Dimensionen
Data_n    = size(Data,2);
Data_m    = size(Data,1);
Data0_n   = size(Data0,2);
Data0_m   = size(Data0,1);
Data1_n   = size(Data1,2);
Data1_m   = size(Data1,1);

% Aufgabe 1

% Aufgabe 2

figure('NumberTitle','off','Name','Aufgabe 2 - Bildpunkte');
hold on
X      = Koordinaten(:,1);
Y      = Koordinaten(:,2);

% Punkte plotten
gscatter(X,Y,Klassen,'krb','+x',[],'off');
hold on

% Diskriminante erzeugen
FDK           = fitcdiscr(Koordinaten,Klassen); % Diskriminantenobjekt - erzeugt Klassifikator
Koeffizienten = FDK.Coeffs(1,2).Const;     % holt die konstanten Koeffizienten vom Klassifikator, die das LGS beschreiben
Linie         = FDK.Coeffs(1,2).Linear;    % holt die linearen Koeffizienten vom Klassifikator, die das LGS beschreiben, also die Grenzlinie

% [x1,x2]*Linie + Koeffizienten = 0
f = @(x1,x2) Linie(1)*x1 + Linie(2)*x2 + Koeffizienten; % y = mx + n

% Funktion plotten
%  Diskriminante = ezplot(f); % dies uebermalt den scatter plot
min_x = min(A(:,1));
max_x = max(A(:,1));
Diskriminante = ezplot(f, [min_x,max_x]); % dies uebermalt den scatter plot nicht, ist aber als Diskriminante falsch
Diskriminante.Color = 'b';

%  xxx = [0 200]; yyy = [0 150]; plot(xxx, yyy); %meins, funktioniert

xlabel('X-Koordinaten');
ylabel('Y-Koordinaten');

%%%% So. Das war das Plotten. Nun kommt die Berechnung:

% scatter within class 1: S1 = \sum(xi - mean)*(xi - mean)'
% scatter within: S_w = S1 + S1
% Gerade w:       w = inv(S_w) * (m1 - m2)

mean1 = mean(Koordinaten0);
mean2 = mean(Koordinaten1);

S1 = 0;
for i = 1:size(Koordinaten0, 1)
    S1 = S1 + (Koordinaten0(i,:) - mean1) * (Koordinaten0(i,:) - mean1)';  % Koordinaten0(i,:). der Punkt sorgt fuer komponentenweise Operation
end

S2 = 0;
for i = 1:size(Koordinaten1, 1)
    S2 = S2 + (Koordinaten1(i,:) - mean2) * (Koordinaten1(i,:) - mean2)';  % Koordinaten0(i,:). der Punkt sorgt fuer komponentenweise Operation
end

S_w = S1 + S1

