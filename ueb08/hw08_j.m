% Clean up
clear all
close all
clc

%%% Aufgabe 1 - Repraesentierbarkeit Boolescher Funktionen durch ein Perzeptron %%%

% Aufgabe 1a

v1 = [0.3; 0.4; 0.5]
v2 = [-0.2; 0.8; 0.3]
v3 = [1; 1; 0]
%  v4 = [0.5; 0.5; -0.8]  % AND

classify_vec(v1)
classify_vec(v2)
classify_vec(v3)


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

