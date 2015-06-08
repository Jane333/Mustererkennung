% Clean up
clear all
close all
clc

% Datenaufbreitung
Data         = load('fisher.txt');
Data0        = Data((Data(:,3)==0),:);
Data1        = Data((Data(:,3)==1),:);
Koordinaten  = Data(:,1:2);
Koordinaten0 = Data((Data(:,3)==0),1:2);
Koordinaten1 = Data((Data(:,3)==1),1:2);
Klassen      = Data(:,3);

% Aufgabe 2

% Grafik erstellen
figure('NumberTitle','off','Name','Aufgabe 2 - Bildpunkte');
    
X = Koordinaten(:,1);
Y = Koordinaten(:,2);
min_x = -250;
max_x =  250;
li = min_x:max_x;
    
% Punkte plotten
gscatter(X,Y,Klassen,'krb','+x',[],'off');
hold on

% Diskriminante erzeugen
FDK    = fitcdiscr(Koordinaten,Klassen);
Konst  = FDK.Coeffs(1,2).Const;
Linear = FDK.Coeffs(1,2).Linear;
fd = @(x1,x2) Linear(1)*x1 + Linear(2)*x2 + Konst;

% Fisher-Diskriminante plotten
Diskriminante = ezplot(fd, [min_x,max_x]);
Diskriminante.Color = 'b';
    
% Scatter within berechnen
mean1 = mean(Koordinaten0);
mean2 = mean(Koordinaten1);
S1 = 0;
for i = 1:size(Koordinaten0, 1)
    S1 = S1 + (Koordinaten0(i,:) - mean1)' * (Koordinaten0(i,:) - mean1);
end
S2 = 0;
for i = 1:size(Koordinaten1, 1)
    S2 = S2 + (Koordinaten1(i,:) - mean2)' * (Koordinaten1(i,:) - mean2);
end
S_w = S1 + S2;
  
% Vektor w berechnen:
w = inv(S_w) * (mean1 - mean2)'  %  w = [0.0019 -0.0019]
w_norm = w / norm(w)
    
% Gerade durch den Vektor w legen und plotten
w_gerade_x = w_norm(1) * li;
w_gerade_y = w_norm(2) * li;
plot(w_gerade_y, w_gerade_x, 'g');    
    
% Daten auf Vektor w_norm projizieren
Koordinaten0_p = [];
for i = 1:size(Koordinaten0, 1)
    Koordinaten0_p = vertcat(Koordinaten0_p, Koordinaten0(i, :) * w_norm);
end
Koordinaten1_p = [];
for i = 1:size(Koordinaten1, 1)
    Koordinaten1_p = vertcat(Koordinaten1_p, Koordinaten1(i, :) * w_norm);
end
Koordinaten_p = vertcat(Koordinaten0_p, Koordinaten1_p);

% pdf der projizierten Daten aus Klasse 1 erzeugen
mean1_p = mean(Koordinaten0_p);
std1_p  = std(Koordinaten0_p);
pdf1_p  = pdf('Normal',li,mean1_p, std1_p);

% pdf der projizierten Daten aus Klasse 2 erzeugen
mean2_p = mean(Koordinaten1_p);
std2_p  = std(Koordinaten1_p);
pdf2_p  = pdf('Normal',li,mean2_p, std2_p);

% Schnittpunkt berechnen
[ispt_x,ispt_y] = intersections(li, pdf1_p, li, pdf2_p, 1);
    
% w0 berechnen und plotten
w0 = ispt_x * (w_norm') %  w0 = [13.6982  -13.8648]
plot(w0(1),w0(2),'m.','markersize',20);
    
% Titel, Bezeichner und Legende der Grafk
title('Aufgabe 2 - Plot');
xlabel('X-Koordinaten');
ylabel('Y-Koordinaten');
legend('1. Klasse','2. Klasse', 'Diskriminante', 'Gerade durch Vektor w', 'Punkt w_0')
axis([-50 300 -50 300])