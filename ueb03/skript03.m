% Uebung 03
% ----------
% Video zu PCA: https://www.youtube.com/watch?v=7OAs0h0kYmk
%
% J. Cavojska, N. Lehmann

% Trainingsdaten, Testdaten und Clusterdaten laden
A = load('pendigits-training.txt');
B = load('pendigits-testing.txt');
C = load('clusters.txt');

% Daten aufbereiten
A_n   = size(A,2);
A_m   = size(A,1);
A = sortrows(A,A_n);
A_0 = A((A(:,17)==0),:);
A_1 = A((A(:,17)==1),:);
A_2 = A((A(:,17)==2),:);
A_3 = A((A(:,17)==3),:);
A_4 = A((A(:,17)==4),:);
A_5 = A((A(:,17)==5),:);
A_6 = A((A(:,17)==6),:);
A_7 = A((A(:,17)==7),:);
A_8 = A((A(:,17)==8),:);
A_9 = A((A(:,17)==9),:);
X = A(:,1:A_n -1);
x = min(X):max(X);

% Aufgabe 1 (3 Punkte)

% A 1.1: multivariate (mehrdimensionale) Normalverteilung
%        (Erwartungswert, Kovarianzmatrix) berechnen

% Erwartungswert für jede Koordinate für jeden Zug (0 bis 9)
E_A_0 = mean(A_0(:,1:A_n -1));
E_A_1 = mean(A_1(:,1:A_n -1));
E_A_2 = mean(A_2(:,1:A_n -1));
E_A_3 = mean(A_3(:,1:A_n -1));
E_A_4 = mean(A_4(:,1:A_n -1));
E_A_5 = mean(A_5(:,1:A_n -1));
E_A_6 = mean(A_6(:,1:A_n -1));
E_A_7 = mean(A_7(:,1:A_n -1));
E_A_8 = mean(A_8(:,1:A_n -1));
E_A_9 = mean(A_9(:,1:A_n -1));

% Kovarianzmatrix für jeden Zug (0 bis 9)
CVM_A_0 = cov(A_0(:,1:A_n -1));
CVM_A_1 = cov(A_1(:,1:A_n -1));
CVM_A_2 = cov(A_2(:,1:A_n -1));
CVM_A_3 = cov(A_3(:,1:A_n -1));
CVM_A_4 = cov(A_4(:,1:A_n -1));
CVM_A_5 = cov(A_5(:,1:A_n -1));
CVM_A_6 = cov(A_6(:,1:A_n -1));
CVM_A_7 = cov(A_7(:,1:A_n -1));
CVM_A_8 = cov(A_8(:,1:A_n -1));
CVM_A_9 = cov(A_9(:,1:A_n -1));

% Multivariante PDF generieren für jeden Zug (0 bis 9)
A_0_mvpdf = mvnpdf(A_0(:,1:A_n -1), E_A_0, CVM_A_0);
A_1_mvpdf = mvnpdf(A_1(:,1:A_n -1), E_A_1, CVM_A_1);
A_2_mvpdf = mvnpdf(A_2(:,1:A_n -1), E_A_2, CVM_A_2);
A_3_mvpdf = mvnpdf(A_3(:,1:A_n -1), E_A_3, CVM_A_3);
A_4_mvpdf = mvnpdf(A_4(:,1:A_n -1), E_A_4, CVM_A_4);
A_5_mvpdf = mvnpdf(A_5(:,1:A_n -1), E_A_5, CVM_A_5);
A_6_mvpdf = mvnpdf(A_6(:,1:A_n -1), E_A_6, CVM_A_6);
A_7_mvpdf = mvnpdf(A_7(:,1:A_n -1), E_A_7, CVM_A_7);
A_8_mvpdf = mvnpdf(A_8(:,1:A_n -1), E_A_8, CVM_A_8);
A_9_mvpdf = mvnpdf(A_9(:,1:A_n -1), E_A_9, CVM_A_9);

% A-Priori-Wahrscheinlichkeit für jeden Zug (0 bis 9)
A_x_apriori = 1 / length(unique(A(:,A_n)));

% A-Posteriori-Wahrscheinlichkeit für jeden Zug (0 bis 9)
A_0_aposteriori = A_0_mvpdf * A_x_apriori;
A_1_aposteriori = A_1_mvpdf * A_x_apriori;
A_2_aposteriori = A_2_mvpdf * A_x_apriori;
A_3_aposteriori = A_3_mvpdf * A_x_apriori;
A_4_aposteriori = A_4_mvpdf * A_x_apriori;
A_5_aposteriori = A_5_mvpdf * A_x_apriori;
A_6_aposteriori = A_6_mvpdf * A_x_apriori;
A_7_aposteriori = A_7_mvpdf * A_x_apriori;
A_8_aposteriori = A_8_mvpdf * A_x_apriori;
A_9_aposteriori = A_9_mvpdf * A_x_apriori;

% A 1.2: Testdaten anhand der A-Posteriori-PDF klassifizieren,
%        Konfusionsmatrix und Klassifikationsguete angeben
%        (Annahme: Gleichverteilte A-Priori-Wahrscheinlichkeit
%                  fuer jede Ziffer)

% Aufgabe 2 (4 Punkte)

% A 2.1: Erste Hauptkomponente der Trainingsdaten angeben

% A 2.2: Dimensionsreduzierung mittels PCA,
%        Testdaten klassifizieren mit Bayes Klassifikator
%        (wie in Aufgabe 1)
%        KlassifikationsgÃ¼te fÃ¼r alle Dimensionen angeben

% Aufgabe 3 (3 Punkte)

% A 3.1: k-means auf die Daten clusters.txt anwenden,
%        k-means soll selbst implementiert werden!

% 1. Phase: Clusterzentren waehlen

% 2. Phase: Expectation mit p(x)

% 3. Phase: Maximization mit Erwartungswert und Kovarianzmatrix

% A 3.2: Clusterzentren und Zuordnungen der Punkte
%        der ersten 5 Iterationsschritte mit k=3
%        visualisieren (insgesamt 5 Bilder)