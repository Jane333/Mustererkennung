% Clean up
clear all
close all
clc

%%% Aufgabe 1 - Repraesentierbarkeit Boolescher Funktionen durch ein Perzeptron %%%

% Aufgabe 1a

v1 = [0.3;0.5;-0.4];   
v2 = [-0.8;-0.6;0.5];  
v3 = [0.7;0.6;-1];     

% signature: int classify_vec([float,float,float], {0|-1})
result_of_v1 = classify_vec(v1, 0)   % f10
result_of_v2 = classify_vec(v2, 0)   % f1
result_of_v3 = classify_vec(v3, 0)   % f8


% Aufgabe 1b)

% create a sphere
[X,Y,Z] = sphere(100);
x = [X(:); X(:); X(:)];
y = [Y(:); Y(:); Y(:)];
z = [Z(:); Z(:); Z(:)];

% plot
figure('NumberTitle','off','Name','DIE Kugel');
hold on
mesh(0.99*X,0.99*Y,0.99*Z)

title('Aufgabe 1b - DIE Kugel');
xlabel('X Koordinaten');
ylabel('Y Koordinaten');
zlabel('Z Koordinaten');
axis([-1.1 1.1 -1.1 1.1]);

% 10000 random vectors
rv = random_vec(10000);

% plot
scatter3(rv(:,1), rv(:,2), rv(:,3), '.', 'm');


% Aufgabe 1c)

% classify random chosen vectors
classifications1 = [];
for i = 1:10000
    v = rv(i,:);
    classifications1 = vertcat(classifications1, classify_vec(v, 0)); % classify v using 0, 1 as boolean values
end

% plot
figure('NumberTitle','off','Name','Histogram of Boolean Function Frequencies');
histogram(classifications1)

tabulate(classifications1)

frequencies = tabulate(classifications1);
maxfreq     = max(frequencies(:,2))
minfreq     = min(frequencies(:,2))
relation    = maxfreq / minfreq


% Aufgabe 1d)

classifications2 = [];
for i = 1:10000
    v = rv(i,:);
    classifications2 = vertcat(classifications2, classify_vec(v, -1));  % classify v using -1, 1 as boolean values
end

% plot
figure('NumberTitle','off','Name','Histogram of -1, 1 Boolean Frequencies');
histogram(classifications2)

tabulate(classifications2)

frequencies = tabulate(classifications2);
maxfreq     = max(frequencies(:,2))
minfreq     = min(frequencies(:,2))
relation    = maxfreq / minfreq