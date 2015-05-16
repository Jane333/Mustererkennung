% Uebung 03
% ----------
% Video zu PCA: https://www.youtube.com/watch?v=7OAs0h0kYmk
%
% J. Cavojska, N. Lehmann

% Trainingsdaten, Testdaten und Clusterdaten laden
A = load('pendigits-training.txt');
B = load('pendigits-testing.txt');
C = load('clusters.txt');

% Aufgabe 1 (3 Punkte)

% A 1.1: multivariate (mehrdimensionale) Normalverteilung
%        (Erwartungswert, Kovarianzmatrix) berechnen

% A 1.2: Testdaten anhand der A-Posteriori-PDF klassifizieren,
%        Konfusionsmatrix und Klassifikationsgüte angeben
%        (Annahme: Gleichverteilte A-Priori-Wahrscheinlichkeit
%                  für jede Ziffer)

% Aufgabe 2 (4 Punkte)

% A 2.1: Erste Hauptkomponente der Trainingsdaten angeben

% A 2.2: Dimensionsreduzierung mittels PCA,
%        Testdaten klassifizieren mit Bayes Klassifikator
%        (wie in Aufgabe 1)
%        Klassifikationsgüte für alle Dimensionen angeben

% Aufgabe 3 (3 Punkte)

% A 3.1: k-means auf die Daten clusters.txt anwenden,
%        k-means soll selbst implementiert werden!

% A 3.2: Clusterzentren und Zuordnungen der Punkte
%        der ersten 5 Iterationsschritte mit k=3
%        visualisieren (insgesamt 5 Bilder)