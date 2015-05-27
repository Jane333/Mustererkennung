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

%  Schaetzen Sie den Wert für “length�? anhand der Parameter “age�? und “temperature�? mit linearer Regression. 
%  Visualisieren Sie dreidimensional die tatsächlichen Datenpunkte, die geschätzten Datenpunkte, die Ebene 
%  auf die projiziert wurde, sowie die Abstände der tatsächlichen Datenpunkte zu dieser Ebene .  

y = A(:,4);  % nur die Laengen der Fische
X = A(:,2:3);  % alle Datenpunkte, ausser Laenge
onesVector = ones(size(X,1), 1); % Spaltenvektor mit Einsen der gleichen Laenge wie A
X = horzcat(onesVector, X);  % Einsen-Vektor an Datenpunkte-Matrix drankleben

% X' ist die transponierte Matrix X
beta = inv(X'*X) * X' * y  % beta ist der Vektor mit den Koeffizienten der Regressionsebene
% Resultat:
% beta =
%     1.0e+03 *
%  
%      3.9043
%      0.0262
%     -0.1064

% TODO: Plotten
fancy_beat = horzcat(A(:,2:3),X*beta);
figure('NumberTitle','off','Name','Aufgabe 1 - Mesh - Surf');

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