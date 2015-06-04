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

% Aufgabe 2
%  
%  figure('NumberTitle','off','Name','Aufgabe 2 - Bildpunkte');
%  hold on
%  X      = Koordinaten(:,1);
%  Y      = Koordinaten(:,2);
%  
%  % Punkte plotten
%  gscatter(X,Y,Klassen,'krb','+x',[],'off');
%  hold on
%  
%  % Diskriminante erzeugen
%  FDK           = fitcdiscr(Koordinaten,Klassen); % Diskriminantenobjekt - erzeugt Klassifikator
%  Koeffizienten = FDK.Coeffs(1,2).Const;     % holt die konstanten Koeffizienten vom Klassifikator, die das LGS beschreiben
%  Linie         = FDK.Coeffs(1,2).Linear;    % holt die linearen Koeffizienten vom Klassifikator, die das LGS beschreiben, also die Grenzlinie
%  
%  % [x1,x2]*Linie + Koeffizienten = 0
%  f = @(x1,x2) Linie(1)*x1 + Linie(2)*x2 + Koeffizienten; % y = mx + n
%  
%  % Funktion plotten
%  %  Diskriminante = ezplot(f); % dies uebermalt den scatter plot
%  min_x = min(A(:,1));
%  max_x = max(A(:,1));
%  Diskriminante = ezplot(f, [min_x,max_x]); % dies uebermalt den scatter plot nicht, ist aber als Diskriminante falsch
%  Diskriminante.Color = 'b';
%  
%  %  xxx = [0 200]; yyy = [0 150]; plot(xxx, yyy); %meins, funktioniert
%  
%  xlabel('X-Koordinaten');
%  ylabel('Y-Koordinaten');


%%%% So. Das war das Plotten. Nun kommt die Berechnung:
% Notiz: Wir nehmen der Einfachheit halber (weil es diese konkrete Aufgabe loest) an, dass sowohl mean1_p und mean2_p als auch std1_p und std2_p jeweils unterschiedlich sind.

mean1 = mean(Koordinaten0);
mean2 = mean(Koordinaten1);

% scatter within class 1: S1 = \sum(x1i - mean1)*(x1i - mean1)'
S1 = 0;
for i = 1:size(Koordinaten0, 1)
    S1 = S1 + (Koordinaten0(i,:) - mean1) * (Koordinaten0(i,:) - mean1)';  % Koordinaten0(i,:). der Punkt sorgt fuer komponentenweise Operation
end

% scatter within class 2: S2 = \sum(x2i - mean2)*(x2i - mean2)'
S2 = 0;
for i = 1:size(Koordinaten1, 1)
    S2 = S2 + (Koordinaten1(i,:) - mean2) * (Koordinaten1(i,:) - mean2)';
end

% scatter within: S_w = S1 + S1
S_w = S1 + S1

% Gerade w, auf die wir projizieren werden: w = (S_w^(-1)) * (m1 - m2)
w = (S_w^(-1)) * (mean1 - mean2)
wn = w / norm(w)  % normalisierte Gerade w

% Daten aus Klasse 1 auf die Gerade w projizieren:
Koordinaten0_p = []  % projizierte Daten aus Klasse 1
for i = 1:size(Koordinaten0, 1)
    Koordinaten0_p = vertcat(Koordinaten0_p, Koordinaten0(i, :) * (wn'));
end

% Daten aus Klasse 2 auf die Gerade w projizieren:
Koordinaten1_p = []  % projizierte Daten aus Klasse 2
for i = 1:size(Koordinaten1, 1)
    Koordinaten1_p = vertcat(Koordinaten1_p, Koordinaten1(i, :) * (wn'));
end
Koordinaten_p = vertcat(Koordinaten0_p, Koordinaten1_p);

% pdf der projizierten Daten aus Klasse 1:
mean1_p = mean(Koordinaten0_p)
std1_p = std(Koordinaten0_p)
xrange1 = min(Koordinaten_p(:,1)):max(Koordinaten_p(:,1)); % Abschnitt auf der x-Achse, fuer den die pdf berechnet werden soll
pdf1_p = pdf('Normal',xrange1,mean1_p, std1_p); % pdf(Art von Verteilung, Abschnitt auf x-Achse, mean, std)

% pdf der projizierten Daten aus Klasse 2:
mean2_p = mean(Koordinaten1_p)
std2_p = std(Koordinaten1_p)
xrange2 = min(Koordinaten_p(:,1)):max(Koordinaten_p(:,1)); % Abschnitt auf der x-Achse, fuer den die pdf berechnet werden soll
pdf2_p = pdf('Normal',xrange2,mean2_p, std2_p); % pdf(Art von Verteilung, Abschnitt auf x-Achse, mean, std)



%  % Ein Versuch, den Schnittpunkt beider pdf-Funktionen mit der p-q-Formel berechnen:
%  p = (2 * mean2_p * std1_p^2 - 2 * mean1_p * std2_p^2) / (std2_p^2 - std1_p^2)
%  q = (2 * std1_p^2 * std2_p^2 * log(std2_p / std1_p) + mean2_p^2 * std1_p^2 - mean1_p^2 * std2_p^2) / (std2_p^2 - std1_p^2)  % log(X) returns the natural log. of X
%  intersection_x = (((-1) * p) / 2) + sqrt((p/2)^2 - q) % dies ist natuerlich totaler Quatsch, x1 und x2 sind keine Koordinaten des Schnittpunktes... oder?
%  intersection_y = (((-1) * p) / 2) - sqrt((p/2)^2 - q)
%  intersection = [intersection_x intersection_y] % this has to be a vector with 2 columns and 1 row 
%  % w0 finden als Projektion des Punktes intersection auf die Gerade w:
%  w0 = intersection * (wn')



% Das ganze doch lieber mit intersections berechnen und plotten:
%  [xout,yout] = intersections(t,y,t,y2,1);
[xout,yout] = intersections(xrange1, pdf1_p, xrange2, pdf2_p, 1);
plot(xrange1, pdf1_p, 'linewidth', 2)
set(gca,'xlim',[min(Koordinaten_p) max(Koordinaten_p)],'ylim',[-1.1 1.1])
hold on
plot(xrange2, pdf2_p, 'g', 'linewidth', 2)
plot(xout,yout,'r.','markersize',18)

intersection = [xout yout]
w0 = intersection * (wn')

% TODO: w und w0 plotten:
