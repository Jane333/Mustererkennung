% Clean up
clear all
close all
clc

%%% Aufgabe 1 - Repräsentierbarkeit Boolescher Funktionen durch ein Perzeptron %%%

b = [ 0,0,1 ; 0,1,1 ; 1,0,1 ; 1,1,1 ];

% Aufgabe 1a)


% Aufgabe 1b)
figure
hold on
[X,Y,Z] = sphere(100);
x = [X(:); X(:); X(:)];
y = [Y(:); Y(:); Y(:)];
z = [Z(:); Z(:); Z(:)];
scatter3(x,y,z, '.', 'm')
scatter3(0,0,0, 'o', 'k', 'filled')
title('Aufgabe 1b - DIE Kugel');
xlabel('X Koordinate');
ylabel('Y Koordinate');
zlabel('Z Koordinate');
axis([-1.1 1.1 -1.1 1.1]);
legend('fancy pinky points');
surf(0.99*X,0.99*Y,0.99*Z)

% Vektoren
Ortsvektoren = [x,y,z]';

% Aufgabe 1c)


% Aufgabe 1d)

