% Clean up
clear all
close all
clc

%%% Aufgabe 1 - Repraesentierbarkeit Boolescher Funktionen durch ein Perzeptron %%%

% Aufgabe 1a

v1 = [0.3; 0.5; -0.4];   % f10
v2 = [-0.8; -0.6; 0.5];  % f1
v3 = [0.7; 0.6; -1];     % f8

classify_vec(v1, 0) % classify v1 using 0, 1 as weights
classify_vec(v2, 0)
classify_vec(v3, 0)


% Aufgabe 1b)
 figure
hold on
[X,Y,Z] = sphere(100);
%  x = [X(:); X(:); X(:)];
%  y = [Y(:); Y(:); Y(:)];
%  z = [Z(:); Z(:); Z(:)];
%  %  scatter3(x,y,z, '.', 'm')
%  scatter3(0,0,0, 'o', 'k', 'filled')
%  title('Aufgabe 1b - DIE Kugel');
%  xlabel('X Koordinate');
%  ylabel('Y Koordinate');
%  zlabel('Z Koordinate');
%  axis([-1.1 1.1 -1.1 1.1]);
%  legend('fancy pinky points');
%  surf(0.99*X,0.99*Y,0.99*Z)
%  
%  % now randomly choose 10000 points out of the ca. 30000 points that currently make up the sphere surface:
%  % Vektoren
%  Ortsvektoren = [x,y,z]; % jede Zeile enthaelt ein x, y, z Tupel
%  randomIndices = randi([1,30603], 1, 10000);
%  while length(unique(randomIndices)) < 10000
%      r = randi([1,30603], 1, 1);
%      randomIndices = horzcat(randomIndices, r);
%  end
%  randomIndices = unique(randomIndices);
%  
%  choice = Ortsvektoren(randomIndices, :); % choose 10000 random Ortsvektoren from the 30603 Ortsvektoren we generated in 1b)
%  scatter3(choice(:,1), choice(:,2), choice(:,3), '.', 'm'); % plot the 10000 randomly chosen points



U = rand(10000, 1) % returns an x times y matrix of uniformly distributed random numbers between 0 and 1
V = rand(10000, 1);
theta = 2 * pi * U
phi = (cos(2*V - 1)).^-1 % cos supports broadcasting, ^ doesn't, hence the .^ to make it element-wise

X2 = sqrt(1 - U.^2) .* cos(theta)
Y2 = sqrt(1 - U.^2) .* sin(theta);
Z2 = U;

choice = [X2, Y2, Z2];
hold on
scatter3(X2, Y2, Z2, '.', 'm'); % plot the 10000 randomly chosen points




% Aufgabe 1c)

% classify all of the 10000 randomly chosen Ortsvektoren:
classifications = [];
for i = 1:10000
    v = choice(i,:);
    classifications = vertcat(classifications, classify_vec(v, 0)); % classify v using 0, 1 as weights
end
figure('NumberTitle','off','Name','Histogram of Boolean Function Frequencies');
histogram(classifications) % hier entsteht ein plot

tabulate(classifications) % creates a frequency table of data in vector classifications
%  Value    Count   Percent
%    0     3490     34.90%
%    1      268      2.68%
%    2      247      2.47%
%    3      391      3.91%
%    4      243      2.43%
%    5      328      3.28%
%    7      128      1.28%
%    8      133      1.33%
%   10      312      3.12%
%   11      234      2.34%
%   12      341      3.41%
%   13      219      2.19%
%   14      260      2.60%
%   15     3406     34.06%
frequencies = tabulate(classifications);
maxfreq = max(frequencies(:,2))  % 3490
minfreq = min(frequencies(:,2))  % 128
verhaeltnis = maxfreq / minfreq  % 27.2656


% Aufgabe 1d)

classifications2 = [];
for i = 1:10000
    v = choice(i,:);
    classifications2 = vertcat(classifications2, classify_vec(v, -1)); % classify v using -1, 1 as weights
end
figure('NumberTitle','off','Name','Histogram of -1, 1 Boolean Frequencies');
histogram(classifications2) % hier entsteht ein plot

tabulate(classifications2) % creates a frequency table of data in vector classifications2
%  Value    Count   Percent
%    0     2192     21.92%
%    1      343      3.43%
%    2      362      3.62%
%    3      801      8.01%
%    4      362      3.62%
%    5      698      6.98%
%    7      337      3.37%
%    8      352      3.52%
%   10      713      7.13%
%   11      322      3.22%
%   12      715      7.15%
%   13      320      3.20%
%   14      327      3.27%
%   15     2156     21.56%
frequencies = tabulate(classifications2);
maxfreq = max(frequencies(:,2))  % 2192
minfreq = min(frequencies(:,2))  % 320
verhaeltnis = maxfreq / minfreq  % 6.8500
