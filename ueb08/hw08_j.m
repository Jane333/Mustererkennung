% Clean up
clear all
close all
clc

%%% Aufgabe 1 - Repraesentierbarkeit Boolescher Funktionen durch ein Perzeptron %%%

% Aufgabe 1a

v1 = [0.3; 0.5; -0.4]   % f10
v2 = [-0.8; -0.6; 0.5]  % f1
v3 = [0.7; 0.6; -1]     % f8


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
Ortsvektoren = [x,y,z]; % jede Zeile enthaelt ein x, y, z Tupel

randomIndices = randi([1,30603], 1, 10000);
while length(unique(randomIndices)) < 10000
    r = randi([1,30603], 1, 1);
    randomIndices = horzcat(randomIndices, r);
end
randomIndices = unique(randomIndices);

choice = Ortsvektoren(randomIndices, :); % choose 10000 random Ortsvektoren from the 30603 Ortsvektoren we generated in 1b)

classifications = [];
for i = 1:10000
    v = choice(i,:);
    classifications = vertcat(classifications, classify_vec(v));
end
myhistogram = hist(classifications);

tabulate(classifications) % creates a frequency table of data in vector classifications

% Aufgabe 1c)


% Aufgabe 1d)

