% Clean up
clear all
close all
clc

%%%%%%%%%%%%%%%%%  Datenaufbereitung  %%%%%%%%%%%%%%%%

% fish.txt: index; the age of the fish; the water temperature in Celsius; the length of the fish
A = load('fish.txt');
A_n   = size(A,2);
A_m   = size(A,1);

% winequality-red.txt: fixed acidity; volatile acidity; citric acid; residual sugar; chlorides; free sulfur dioxide; total sulfur; dioxide; density; pH; sulphates; alcohol; quality (score between 0 and 10) 
B = load('winequality-red.txt');
B_n   = size(B,2);
B_m   = size(B,1);


%%%%%%%%%%%%%%%%%%%%  AUFGABE 1  %%%%%%%%%%%%%%%%%%%%%

%  Schaetzen Sie den Wert fuer length anhand der Parameter age und temperature mit linearer Regression. 
%  Visualisieren Sie dreidimensional die tatsaechlichen Datenpunkte, die geschaetzten Datenpunkte, die Ebene 
%  auf die projiziert wurde, sowie die Abstaende der tatsaechlichen Datenpunkte zu dieser Ebene.  

y = A(:,4);  % nur die Laengen der Fische
Z = A(:,2:3);  % alle Datenpunkte, ausser Laenge
onesVector = ones(size(Z,1), 1); % Spaltenvektor mit Einsen der gleichen Laenge wie A
X = horzcat(onesVector, Z);  % Einsen-Vektor an Datenpunkte-Matrix drankleben

% X' ist die transponierte Matrix X
beta = inv(X'*X) * X' * y  % beta ist der Vektor mit den Koeffizienten der Regressionsebene
% Resultat:
% beta =
%     1.0e+03 *
%  
%      3.9043
%      0.0262
%     -0.1064

% Ebene berechnen
Xbeta = X*beta; % rechnet die geschaetzten Werte fuer lenght aus
E = horzcat(Z, Xbeta); % nur zum Ansehen: linear regressierte Daten
E = horzcat(E,y); % nur zum Vergleich: Ursprungsdaten

% Skalierung ueber Extrema? (14-153)
minE1 = min(E(:,1)');
maxE1 = max(E(:,1)');
minE2 = min(E(:,2)');
maxE2 = max(E(:,2)');
minE = min(minE1,minE2);
maxE = max(maxE1,maxE2);

% Ebene plotten
figure('NumberTitle','off','Name','Aufgabe 1 - Mesh');

x=minE:maxE; 
y=minE:maxE; 
Z=beta(1,1)*x+beta(2,1)*y+beta(3,1); 
Z = repmat(Z,maxE,1); 
mesh(Z); % die Ebene


% die Bildpunkte muessen jetzt noch in das Bild...
bildPunkte = A(:,2:4);
hold on
% plotting the data points into the same figure:
scatter3(bildPunkte(:,1), bildPunkte(:,2), bildPunkte(:,3), 60, [1 0 0])
%hold on
% plotting the lines connecting data points and plane into the same figure:

xlabel('Alter'); 
ylabel('Wassertemperatur'); 
zlabel('Laenge');
legend('Linear regressierte Fischlaenge')

%%%%%%%%%%%%%%%%%%%%  AUFGABE 2  %%%%%%%%%%%%%%%%%%%%%

% Schaetzen Sie den Wert fuer quality mit linearer Regression anhand aller moeglichen Kombinationen der anderen Parameter (also jeweils für alle Einer­, Zweier­, Dreierkombinationen, usw.) und berechnen jeweils die Summe der quadratischen Abweichungen zwischen den geschätzten und tatsächlichen Werten für “quality�?.
% Visualisieren Sie dies als zweidimensionalen Plot. Auf der X�Achse steht dabei die Anzahl der verwendeten Parameter, auf der y­Achse die Summe der quadratischen Abweichungen.
 
y = B(:,12);  % Spalte mit Weinqualitaet
Result = [];  % Ergebnismatrix

featureIndices = [1 2 3 4 5 6 7 8 9 10 11];
for numFeatures = 1:11  % es gibt 11 features, anhand welcher man klassifizieren kann
    combinations = combnk(featureIndices, numFeatures);
    % for combination = combinations
    for line = 1:size(combinations, 1)
        combination = combinations(line,:);
        X = B(:, combination);
        onesVector = ones(size(X,1), 1); % Spaltenvektor mit Einsen der gleichen Laenge wie B
        X = horzcat(onesVector, X);      % Einsen-Vektor an Datenpunkte-Matrix drankleben
        beta = inv(X'*X) * X' * y;       % beta ist der Vektor mit den Koeffizienten der Regressionsebene
        
        Q = (y - X*beta)'*(y - X*beta);  % mean squared error
        Result = vertcat(Result, [numFeatures, Q]);
    end
end % for numFeatures

figure('NumberTitle','off','Name','Aufgabe 2 - Scatter');
scatter(Result(:,1),Result(:,2));