%%%%%%%%%%%%%%%%%  Datenaufbereitung  %%%%%%%%%%%%%%%%

% fish.txt: index ; the age of the fish ; the water temperature in degrees Celsius ; the length of the fish
A = load('fish.txt');
A_n   = size(A,2);
A_m   = size(A,1);

% winequality-red.txt: fixed acidity ; volatile acidity ; citric acid ; residual sugar ; chlorides ; free sulfur dioxide ; total sulfur ; dioxide ; density ; pH ; sulphates ; alcohol ; quality (score between 0 and 10) 
B = load('winequality-red.txt');
B_n   = size(B,2);
B_m   = size(B,1);


%%%%%%%%%%%%%%%%%%%%  AUFGABE 1  %%%%%%%%%%%%%%%%%%%%%

%  Schätzen Sie den Wert für “length” anhand der Parameter “age” und “temperature” mit linearer Regression. 
%  Visualisieren Sie dreidimensional die tatsächlichen Datenpunkte, die geschätzten Datenpunkte, die Ebene 
%  auf die projiziert wurde, sowie die Abstände der tatsächlichen Datenpunkte zu dieser Ebene .  

X = A(:,1:2)  % alle Datenpunkte, ausser Laenge
y = A(:,3)  % nur die Laengen der Fische

% X.' ist die transponierte Matrix X
beta = inv(X.'*X) * X' * y  % beta ist ein Vektor mit den Koeffizienten der Regressionsebene