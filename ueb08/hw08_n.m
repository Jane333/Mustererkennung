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
surf(0.99*X,0.99*Y,0.99*Z)
scatter3(x,y,z, '.', 'm')
title('Aufgabe 1b - DIE Kugel');
xlabel('X Koordinaten');
ylabel('Y Koordinaten');
zlabel('Z Koordinaten');
axis([-1.1 1.1 -1.1 1.1]);

% randomly choose 10000 vectors out of the 30603 sphere vectors
vectors = [x,y,z];
randomIndices = randi([1,30603], 1, 10000);
while length(unique(randomIndices)) < 10000
    r = randi([1,30603], 1, 1);
    randomIndices = horzcat(randomIndices, r);
end
randomIndices = unique(randomIndices);

% make the choice
choice = vectors(randomIndices, :);

% plot
scatter3(choice(:,1), choice(:,2), choice(:,3), '.', 'm');


% Aufgabe 1c)

% classify random chosen vectors
classifications1 = [];
for i = 1:10000
    v = choice(i,:);
    classifications1 = vertcat(classifications1, classify_vec(v, 0)); % classify v using 0, 1 as weights
end

% plot
figure('NumberTitle','off','Name','Histogram of Boolean Function Frequencies');
histogram(classifications1)

tabulate(classifications1)
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

frequencies = tabulate(classifications1);
maxfreq     = max(frequencies(:,2))       % 3490
minfreq     = min(frequencies(:,2))       % 128
relation    = maxfreq / minfreq           % 27.2656


% Aufgabe 1d)

classifications2 = [];
for i = 1:10000
    v = choice(i,:);
    classifications2 = vertcat(classifications2, classify_vec(v, -1));
end

% plot
figure('NumberTitle','off','Name','Histogram of -1, 1 Boolean Frequencies');
histogram(classifications2)

tabulate(classifications2)
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
maxfreq     = max(frequencies(:,2))        % 2192
minfreq     = min(frequencies(:,2))        % 320
relation    = maxfreq / minfreq            % 6.8500