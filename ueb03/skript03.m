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
A_nl = A(:,1:A_n -1);
A_0 = A((A(:,17)==0),:);
A_0_nl = A_0(:,1:A_n -1);
A_1 = A((A(:,17)==1),:);
A_1_nl = A_1(:,1:A_n -1);
A_2 = A((A(:,17)==2),:);
A_2_nl = A_2(:,1:A_n -1);
A_3 = A((A(:,17)==3),:);
A_3_nl = A_3(:,1:A_n -1);
A_4 = A((A(:,17)==4),:);
A_4_nl = A_4(:,1:A_n -1);
A_5 = A((A(:,17)==5),:);
A_5_nl = A_5(:,1:A_n -1);
A_6 = A((A(:,17)==6),:);
A_6_nl = A_6(:,1:A_n -1);
A_7 = A((A(:,17)==7),:);
A_7_nl = A_7(:,1:A_n -1);
A_8 = A((A(:,17)==8),:);
A_8_nl = A_8(:,1:A_n -1);
A_9 = A((A(:,17)==9),:);
A_9_nl = A_9(:,1:A_n -1);
X = A(:,1:A_n -1); % alle Daten ausser der Zuglinie
x = min(X):max(X);
B_n   = size(B,2);
B_m   = size(B,1);
B_nl = B(:,1:B_n -1);

% Aufgabe 1 (3 Punkte)

% A 1.1: multivariate (mehrdimensionale) Normalverteilung
%        (Erwartungswert, Kovarianzmatrix) berechnen

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9)
E_A_0 = mean(A_0_nl);
E_A_1 = mean(A_1_nl);
E_A_2 = mean(A_2_nl);
E_A_3 = mean(A_3_nl);
E_A_4 = mean(A_4_nl);
E_A_5 = mean(A_5_nl);
E_A_6 = mean(A_6_nl);
E_A_7 = mean(A_7_nl);
E_A_8 = mean(A_8_nl);
E_A_9 = mean(A_9_nl);

% Kovarianzmatrix fuer jeden Zug (0 bis 9)
CVM_A_0 = cov(A_0_nl);
CVM_A_1 = cov(A_1_nl);
CVM_A_2 = cov(A_2_nl);
CVM_A_3 = cov(A_3_nl);
CVM_A_4 = cov(A_4_nl);
CVM_A_5 = cov(A_5_nl);
CVM_A_6 = cov(A_6_nl);
CVM_A_7 = cov(A_7_nl);
CVM_A_8 = cov(A_8_nl);
CVM_A_9 = cov(A_9_nl);

% Multivariante PDF generieren fuer jeden Zug (0 bis 9)
% wir geben hier kein Intervall an, weil die pdf hochdimensional ist 
% und nicht nur fuer einen bestimmten Bereich berechnet werden soll
A_0_mvpdf = mvnpdf(A_0_nl, E_A_0, CVM_A_0);
A_1_mvpdf = mvnpdf(A_1_nl, E_A_1, CVM_A_1);
A_2_mvpdf = mvnpdf(A_2_nl, E_A_2, CVM_A_2);
A_3_mvpdf = mvnpdf(A_3_nl, E_A_3, CVM_A_3);
A_4_mvpdf = mvnpdf(A_4_nl, E_A_4, CVM_A_4);
A_5_mvpdf = mvnpdf(A_5_nl, E_A_5, CVM_A_5);
A_6_mvpdf = mvnpdf(A_6_nl, E_A_6, CVM_A_6);
A_7_mvpdf = mvnpdf(A_7_nl, E_A_7, CVM_A_7);
A_8_mvpdf = mvnpdf(A_8_nl, E_A_8, CVM_A_8);
A_9_mvpdf = mvnpdf(A_9_nl, E_A_9, CVM_A_9);


% A 1.2: Testdaten anhand der A-Posteriori-PDF klassifizieren,
%        Konfusionsmatrix und Klassifikationsguete angeben
%        (Annahme: Gleichverteilte A-Priori-Wahrscheinlichkeit
%                  fuer jede Ziffer)

% A-Priori-Wahrscheinlichkeit fuer jeden Zug (0 bis 9)
A_x_apriori = 1 / length(unique(A(:,A_n)));

% A-Posteriori-Wahrscheinlichkeit fuer jeden Zug (0 bis 9)
% P(Zuglinie | Position) = P(Position | Zuglinie) * P(Zuglinie)
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

% Wir klassifizieren mit der L2-Norm.
M_classify = [];
for index = 1:size(B,1)
    trainData = B(index,1:B_n -1);
    A_0_aposteriori_predict = mvnpdf(trainData, E_A_0, CVM_A_0);
    A_1_aposteriori_predict = mvnpdf(trainData, E_A_1, CVM_A_1);
    A_2_aposteriori_predict = mvnpdf(trainData, E_A_2, CVM_A_2);
    A_3_aposteriori_predict = mvnpdf(trainData, E_A_3, CVM_A_3);
    A_4_aposteriori_predict = mvnpdf(trainData, E_A_4, CVM_A_4);
    A_5_aposteriori_predict = mvnpdf(trainData, E_A_5, CVM_A_5);
    A_6_aposteriori_predict = mvnpdf(trainData, E_A_6, CVM_A_6);
    A_7_aposteriori_predict = mvnpdf(trainData, E_A_7, CVM_A_7);
    A_8_aposteriori_predict = mvnpdf(trainData, E_A_8, CVM_A_8);
    A_9_aposteriori_predict = mvnpdf(trainData, E_A_9, CVM_A_9);
    
    [maxValue, indexAtMaxValue] = max([norm(A_0_aposteriori_predict),norm(A_1_aposteriori_predict),norm(A_2_aposteriori_predict),norm(A_3_aposteriori_predict),norm(A_4_aposteriori_predict),norm(A_5_aposteriori_predict),norm(A_6_aposteriori_predict),norm(A_7_aposteriori_predict),norm(A_8_aposteriori_predict),norm(A_9_aposteriori_predict)]);
    
    if (maxValue == norm(A_0_aposteriori_predict))     % train 0 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),0];
        M_classify = vertcat(M_classify,tmpVector);
    elseif (maxValue == norm(A_1_aposteriori_predict)) % train 1 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),1];
        M_classify = vertcat(M_classify,tmpVector);
    elseif (maxValue == norm(A_2_aposteriori_predict)) % train 2 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),2];
        M_classify = vertcat(M_classify,tmpVector);
    elseif (maxValue == norm(A_3_aposteriori_predict)) % train 3 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),3];
        M_classify = vertcat(M_classify,tmpVector);
    elseif (maxValue == norm(A_4_aposteriori_predict)) % train 4 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),4];
        M_classify = vertcat(M_classify,tmpVector);
    elseif (maxValue == norm(A_5_aposteriori_predict)) % train 5 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),5];
        M_classify = vertcat(M_classify,tmpVector);
    elseif (maxValue == norm(A_6_aposteriori_predict)) % train 6 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),6];
        M_classify = vertcat(M_classify,tmpVector);
    elseif (maxValue == norm(A_7_aposteriori_predict)) % train 7 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),7];
        M_classify = vertcat(M_classify,tmpVector);
    elseif (maxValue == norm(A_8_aposteriori_predict)) % train 8 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),8];
        M_classify = vertcat(M_classify,tmpVector);
    else                                       % train 9 predicted
        tmpVector = [B(index,1:B_n -1),B(index,B_n),9];
        M_classify = vertcat(M_classify,tmpVector);
    end % end-if
end % end-for_each

% Konfusionsmatrix
knownClass = M_classify(:, B_n);
predictedClass = M_classify(:, B_n +1);
confusionmat(knownClass, predictedClass)

%   341     0     0     0     0     0     0     0    22     0
%     0   350    12     0     1     0     0     0     1     0
%     0     8   355     0     0     0     0     1     0     0
%     0     9     0   320     0     1     0     1     0     5
%     0     0     0     0   362     0     0     0     0     2
%     0     0     0     1     0   323     0     0     2     9
%     0     0     0     0     0     0   325     0    11     0
%     0    28     0     0     0     0     0   314     5    17
%     0     0     0     0     0     0     0     0   336     0
%     0     5     0     0     0     0     0     1     1   329

% Klassifikationsguete
M_m = size(M_classify, 1);
corret_predicted = 0;
for index = 1:M_m
    if M_classify(index, B_n) == M_classify(index, B_n +1)
        corret_predicted = corret_predicted + 1;
    end
end
classification_quality = corret_predicted / M_m

%   0.9591

%%%%%%%%%%%%%%%%%%%%%%%%%%  Aufgabe 2 (4 Punkte)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A 2.1: Erste Hauptkomponente der Trainingsdaten angeben

% Schritt 1: Subtraktion der Mittelwerte 
%  M_alt = A(:,1:A_n -1);
%  mu=mean(M_alt')';
%  M_neu = repmat(mu,1,size(M_alt,2));
%  M_minus_mean = M_alt - M_neu;

% Schritt 2: Berechnung der Kovarianzmatrix
CVM_A = cov(A_nl);
CVM_B = cov(B_nl);

% Schritt 3: Eigenwerte und Eigenvektoren der Kovarianzmatrix
[VB,DB] = eig(CVM_A);
EigVec_CVM_A = VB; % Eigenvektoren von CVM_A
EigVal_CVM_A = DB; % Diagonalmatrix der Eigenwerte zu CVM_A

[VB,DB] = eig(CVM_B);
EigVec_CVM_B = VB; % Eigenvektoren von CVM_B
EigVal_CVM_B = DB; % Diagonalmatrix der Eigenwerte zu CVM_B      

erste_hauptkomponente = EigVec_CVM_A(:,1) % die erste Hauptkomponente von CVM_A

% A 2.2: Dimensionsreduzierung mittels PCA,
%        Testdaten klassifizieren mit Bayes Klassifikator
%        (wie in Aufgabe 1)
%        Klassifikationsguete fuer alle Dimensionen angeben

% Unterräume erzeugen
pca_ur_1dim  = EigVec_CVM_A(:,1);    % Unterraum 1 dimensional
pca_ur_2dim  = EigVec_CVM_A(:,1:2);  % Unterraum 2 dimensional
pca_ur_3dim  = EigVec_CVM_A(:,1:3);  % Unterraum 3 dimensional
pca_ur_4dim  = EigVec_CVM_A(:,1:4);  % Unterraum 4 dimensional
pca_ur_5dim  = EigVec_CVM_A(:,1:5);  % Unterraum 5 dimensional
pca_ur_6dim  = EigVec_CVM_A(:,1:6);  % Unterraum 6 dimensional
pca_ur_7dim  = EigVec_CVM_A(:,1:7);  % Unterraum 7 dimensional
pca_ur_8dim  = EigVec_CVM_A(:,1:8);  % Unterraum 8 dimensional
pca_ur_9dim  = EigVec_CVM_A(:,1:9);  % Unterraum 9 dimensional
pca_ur_10dim = EigVec_CVM_A(:,1:10); % Unterraum 10 dimensional
pca_ur_11dim = EigVec_CVM_A(:,1:11); % Unterraum 11 dimensional
pca_ur_12dim = EigVec_CVM_A(:,1:12); % Unterraum 12 dimensional
pca_ur_13dim = EigVec_CVM_A(:,1:13); % Unterraum 13 dimensional
pca_ur_14dim = EigVec_CVM_A(:,1:14); % Unterraum 14 dimensional
pca_ur_15dim = EigVec_CVM_A(:,1:15); % Unterraum 15 dimensional
pca_ur_16dim = EigVec_CVM_A(:,1:16); % Unterraum 16 dimensional

% Abbildung der kompletten Trainingsdaten in einen Unterraum
A_ur_dim1  = A_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_1dim
A_ur_dim2  = A_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_2dim
A_ur_dim3  = A_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_3dim
A_ur_dim4  = A_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_4dim
A_ur_dim5  = A_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_5dim
A_ur_dim6  = A_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_6dim
A_ur_dim7  = A_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_7dim
A_ur_dim8  = A_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_8dim
A_ur_dim9  = A_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_9dim
A_ur_dim10 = A_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_10dim
A_ur_dim11 = A_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_11dim
A_ur_dim12 = A_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_12dim
A_ur_dim13 = A_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_13dim
A_ur_dim14 = A_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_14dim
A_ur_dim15 = A_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_15dim
A_ur_dim16 = A_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 0 in einen Unterraum
A_0_ur_dim1  = A_0_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_1dim
A_0_ur_dim2  = A_0_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_2dim
A_0_ur_dim3  = A_0_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_3dim
A_0_ur_dim4  = A_0_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_4dim
A_0_ur_dim5  = A_0_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_5dim
A_0_ur_dim6  = A_0_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_6dim
A_0_ur_dim7  = A_0_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_7dim
A_0_ur_dim8  = A_0_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_8dim
A_0_ur_dim9  = A_0_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_9dim
A_0_ur_dim10 = A_0_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_10dim
A_0_ur_dim11 = A_0_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_11dim
A_0_ur_dim12 = A_0_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_12dim
A_0_ur_dim13 = A_0_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_13dim
A_0_ur_dim14 = A_0_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_14dim
A_0_ur_dim15 = A_0_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_15dim
A_0_ur_dim16 = A_0_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_0_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 1 in einen Unterraum
A_1_ur_dim1  = A_1_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_1dim
A_1_ur_dim2  = A_1_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_2dim
A_1_ur_dim3  = A_1_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_3dim
A_1_ur_dim4  = A_1_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_4dim
A_1_ur_dim5  = A_1_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_5dim
A_1_ur_dim6  = A_1_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_6dim
A_1_ur_dim7  = A_1_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_7dim
A_1_ur_dim8  = A_1_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_8dim
A_1_ur_dim9  = A_1_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_9dim
A_1_ur_dim10 = A_1_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_10dim
A_1_ur_dim11 = A_1_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_11dim
A_1_ur_dim12 = A_1_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_12dim
A_1_ur_dim13 = A_1_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_13dim
A_1_ur_dim14 = A_1_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_14dim
A_1_ur_dim15 = A_1_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_15dim
A_1_ur_dim16 = A_1_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 2 in einen Unterraum
A_2_ur_dim1  = A_2_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_1dim
A_2_ur_dim2  = A_2_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_2dim
A_2_ur_dim3  = A_2_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_3dim
A_2_ur_dim4  = A_2_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_4dim
A_2_ur_dim5  = A_2_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_5dim
A_2_ur_dim6  = A_2_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_6dim
A_2_ur_dim7  = A_2_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_7dim
A_2_ur_dim8  = A_2_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_8dim
A_2_ur_dim9  = A_2_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_9dim
A_2_ur_dim10 = A_2_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_10dim
A_2_ur_dim11 = A_2_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_11dim
A_2_ur_dim12 = A_2_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_12dim
A_2_ur_dim13 = A_2_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_13dim
A_2_ur_dim14 = A_2_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_14dim
A_2_ur_dim15 = A_2_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_15dim
A_2_ur_dim16 = A_2_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_2_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 3 in einen Unterraum
A_3_ur_dim1  = A_3_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_1dim
A_3_ur_dim2  = A_3_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_2dim
A_3_ur_dim3  = A_3_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_3dim
A_3_ur_dim4  = A_3_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_4dim
A_3_ur_dim5  = A_3_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_5dim
A_3_ur_dim6  = A_3_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_6dim
A_3_ur_dim7  = A_3_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_7dim
A_3_ur_dim8  = A_3_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_8dim
A_3_ur_dim9  = A_3_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_9dim
A_3_ur_dim10 = A_3_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_10dim
A_3_ur_dim11 = A_3_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_11dim
A_3_ur_dim12 = A_3_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_1_nl in den Unterraum pca_ur_12dim
A_3_ur_dim13 = A_3_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_13dim
A_3_ur_dim14 = A_3_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_14dim
A_3_ur_dim15 = A_3_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_15dim
A_3_ur_dim16 = A_3_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_3_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 4 in einen Unterraum
A_4_ur_dim1  = A_4_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_1dim
A_4_ur_dim2  = A_4_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_2dim
A_4_ur_dim3  = A_4_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_3dim
A_4_ur_dim4  = A_4_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_4dim
A_4_ur_dim5  = A_4_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_5dim
A_4_ur_dim6  = A_4_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_6dim
A_4_ur_dim7  = A_4_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_7dim
A_4_ur_dim8  = A_4_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_8dim
A_4_ur_dim9  = A_4_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_9dim
A_4_ur_dim10 = A_4_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_10dim
A_4_ur_dim11 = A_4_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_11dim
A_4_ur_dim12 = A_4_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_12dim
A_4_ur_dim13 = A_4_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_13dim
A_4_ur_dim14 = A_4_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_14dim
A_4_ur_dim15 = A_4_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_15dim
A_4_ur_dim16 = A_4_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_4_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 5 in einen Unterraum
A_5_ur_dim1  = A_5_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_1dim
A_5_ur_dim2  = A_5_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_2dim
A_5_ur_dim3  = A_5_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_3dim
A_5_ur_dim4  = A_5_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_4dim
A_5_ur_dim5  = A_5_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_5dim
A_5_ur_dim6  = A_5_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_6dim
A_5_ur_dim7  = A_5_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_7dim
A_5_ur_dim8  = A_5_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_8dim
A_5_ur_dim9  = A_5_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_9dim
A_5_ur_dim10 = A_5_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_10dim
A_5_ur_dim11 = A_5_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_11dim
A_5_ur_dim12 = A_5_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_12dim
A_5_ur_dim13 = A_5_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_13dim
A_5_ur_dim14 = A_5_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_14dim
A_5_ur_dim15 = A_5_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_15dim
A_5_ur_dim16 = A_5_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_5_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 6 in einen Unterraum
A_6_ur_dim1  = A_6_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_1dim
A_6_ur_dim2  = A_6_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_2dim
A_6_ur_dim3  = A_6_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_3dim
A_6_ur_dim4  = A_6_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_4dim
A_6_ur_dim5  = A_6_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_5dim
A_6_ur_dim6  = A_6_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_6dim
A_6_ur_dim7  = A_6_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_7dim
A_6_ur_dim8  = A_6_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_8dim
A_6_ur_dim9  = A_6_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_9dim
A_6_ur_dim10 = A_6_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_10dim
A_6_ur_dim11 = A_6_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_11dim
A_6_ur_dim12 = A_6_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_12dim
A_6_ur_dim13 = A_6_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_13dim
A_6_ur_dim14 = A_6_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_14dim
A_6_ur_dim15 = A_6_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_15dim
A_6_ur_dim16 = A_6_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_6_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 7 in einen Unterraum
A_7_ur_dim1  = A_7_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_1dim
A_7_ur_dim2  = A_7_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_2dim
A_7_ur_dim3  = A_7_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_3dim
A_7_ur_dim4  = A_7_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_4dim
A_7_ur_dim5  = A_7_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_5dim
A_7_ur_dim6  = A_7_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_6dim
A_7_ur_dim7  = A_7_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_7dim
A_7_ur_dim8  = A_7_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_8dim
A_7_ur_dim9  = A_7_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_9dim
A_7_ur_dim10 = A_7_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_10dim
A_7_ur_dim11 = A_7_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_11dim
A_7_ur_dim12 = A_7_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_12dim
A_7_ur_dim13 = A_7_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_13dim
A_7_ur_dim14 = A_7_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_14dim
A_7_ur_dim15 = A_7_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_15dim
A_7_ur_dim16 = A_7_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_7_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 8 in einen Unterraum
A_8_ur_dim1  = A_8_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_1dim
A_8_ur_dim2  = A_8_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_2dim
A_8_ur_dim3  = A_8_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_3dim
A_8_ur_dim4  = A_8_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_4dim
A_8_ur_dim5  = A_8_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_5dim
A_8_ur_dim6  = A_8_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_6dim
A_8_ur_dim7  = A_8_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_7dim
A_8_ur_dim8  = A_8_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_8dim
A_8_ur_dim9  = A_8_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_9dim
A_8_ur_dim10 = A_8_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_10dim
A_8_ur_dim11 = A_8_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_11dim
A_8_ur_dim12 = A_8_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_12dim
A_8_ur_dim13 = A_8_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_13dim
A_8_ur_dim14 = A_8_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_14dim
A_8_ur_dim15 = A_8_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_15dim
A_8_ur_dim16 = A_8_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_8_nl in den Unterraum pca_ur_16dim

% Abbildung der Trainingsdaten der Zuglinie 9 in einen Unterraum
A_9_ur_dim1  = A_9_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_1dim
A_9_ur_dim2  = A_9_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_2dim
A_9_ur_dim3  = A_9_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_3dim
A_9_ur_dim4  = A_9_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_4dim
A_9_ur_dim5  = A_9_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_5dim
A_9_ur_dim6  = A_9_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_6dim
A_9_ur_dim7  = A_9_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_7dim
A_9_ur_dim8  = A_9_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_8dim
A_9_ur_dim9  = A_9_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_9dim
A_9_ur_dim10 = A_9_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_10dim
A_9_ur_dim11 = A_9_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_11dim
A_9_ur_dim12 = A_9_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_12dim
A_9_ur_dim13 = A_9_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_13dim
A_9_ur_dim14 = A_9_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_14dim
A_9_ur_dim15 = A_9_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_15dim
A_9_ur_dim16 = A_9_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus A_9_nl in den Unterraum pca_ur_16dim

% Abbidung der Testdaten in einen Unterraum
B_ur_dim1  = B_nl * pca_ur_1dim;  % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_1dim
B_ur_dim2  = B_nl * pca_ur_2dim;  % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_2dim
B_ur_dim3  = B_nl * pca_ur_3dim;  % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_3dim
B_ur_dim4  = B_nl * pca_ur_4dim;  % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_4dim
B_ur_dim5  = B_nl * pca_ur_5dim;  % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_5dim
B_ur_dim6  = B_nl * pca_ur_6dim;  % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_6dim
B_ur_dim7  = B_nl * pca_ur_7dim;  % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_7dim
B_ur_dim8  = B_nl * pca_ur_8dim;  % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_8dim
B_ur_dim9  = B_nl * pca_ur_9dim;  % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_9dim
B_ur_dim10 = B_nl * pca_ur_10dim; % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_10dim
B_ur_dim11 = B_nl * pca_ur_11dim; % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_11dim
B_ur_dim12 = B_nl * pca_ur_12dim; % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_12dim
B_ur_dim13 = B_nl * pca_ur_13dim; % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_13dim
B_ur_dim14 = B_nl * pca_ur_14dim; % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_14dim
B_ur_dim15 = B_nl * pca_ur_15dim; % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_15dim
B_ur_dim16 = B_nl * pca_ur_16dim; % Abbildung der Datenpunkte aus B_nl in den Unterraum pca_ur_16dim

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 1 dimensionalen Unterraum
E_A_0_ur_dim1 = mean(A_0_ur_dim1);
E_A_1_ur_dim1 = mean(A_1_ur_dim1);
E_A_2_ur_dim1 = mean(A_2_ur_dim1);
E_A_3_ur_dim1 = mean(A_3_ur_dim1);
E_A_4_ur_dim1 = mean(A_4_ur_dim1);
E_A_5_ur_dim1 = mean(A_5_ur_dim1);
E_A_6_ur_dim1 = mean(A_6_ur_dim1);
E_A_7_ur_dim1 = mean(A_7_ur_dim1);
E_A_8_ur_dim1 = mean(A_8_ur_dim1);
E_A_9_ur_dim1 = mean(A_9_ur_dim1);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 2 dimensionalen Unterraum
E_A_0_ur_dim2 = mean(A_0_ur_dim2);
E_A_1_ur_dim2 = mean(A_1_ur_dim2);
E_A_2_ur_dim2 = mean(A_2_ur_dim2);
E_A_3_ur_dim2 = mean(A_3_ur_dim2);
E_A_4_ur_dim2 = mean(A_4_ur_dim2);
E_A_5_ur_dim1 = mean(A_5_ur_dim2);
E_A_6_ur_dim2 = mean(A_6_ur_dim2);
E_A_7_ur_dim2 = mean(A_7_ur_dim2);
E_A_8_ur_dim2 = mean(A_8_ur_dim2);
E_A_9_ur_dim2 = mean(A_9_ur_dim2);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 3 dimensionalen Unterraum
E_A_0_ur_dim3 = mean(A_0_ur_dim3);
E_A_1_ur_dim3 = mean(A_1_ur_dim3);
E_A_2_ur_dim3 = mean(A_2_ur_dim3);
E_A_3_ur_dim3 = mean(A_3_ur_dim3);
E_A_4_ur_dim3 = mean(A_4_ur_dim3);
E_A_5_ur_dim3 = mean(A_5_ur_dim3);
E_A_6_ur_dim3 = mean(A_6_ur_dim3);
E_A_7_ur_dim3 = mean(A_7_ur_dim3);
E_A_8_ur_dim3 = mean(A_8_ur_dim3);
E_A_9_ur_dim3 = mean(A_9_ur_dim3);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 4 dimensionalen Unterraum
E_A_0_ur_dim4 = mean(A_0_ur_dim4);
E_A_1_ur_dim4 = mean(A_1_ur_dim4);
E_A_2_ur_dim4 = mean(A_2_ur_dim4);
E_A_3_ur_dim4 = mean(A_3_ur_dim4);
E_A_4_ur_dim4 = mean(A_4_ur_dim4);
E_A_5_ur_dim4 = mean(A_5_ur_dim4);
E_A_6_ur_dim4 = mean(A_6_ur_dim4);
E_A_7_ur_dim4 = mean(A_7_ur_dim4);
E_A_8_ur_dim4 = mean(A_8_ur_dim4);
E_A_9_ur_dim4 = mean(A_9_ur_dim4);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 5 dimensionalen Unterraum
E_A_0_ur_dim5 = mean(A_0_ur_dim5);
E_A_1_ur_dim5 = mean(A_1_ur_dim5);
E_A_2_ur_dim5 = mean(A_2_ur_dim5);
E_A_3_ur_dim5 = mean(A_3_ur_dim5);
E_A_4_ur_dim5 = mean(A_4_ur_dim5);
E_A_5_ur_dim5 = mean(A_5_ur_dim5);
E_A_6_ur_dim5 = mean(A_6_ur_dim5);
E_A_7_ur_dim5 = mean(A_7_ur_dim5);
E_A_8_ur_dim5 = mean(A_8_ur_dim5);
E_A_9_ur_dim5 = mean(A_9_ur_dim5);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 6 dimensionalen Unterraum
E_A_0_ur_dim6 = mean(A_0_ur_dim6);
E_A_1_ur_dim6 = mean(A_1_ur_dim6);
E_A_2_ur_dim6 = mean(A_2_ur_dim6);
E_A_3_ur_dim6 = mean(A_3_ur_dim6);
E_A_4_ur_dim6 = mean(A_4_ur_dim6);
E_A_5_ur_dim6 = mean(A_5_ur_dim6);
E_A_6_ur_dim6 = mean(A_6_ur_dim6);
E_A_7_ur_dim6 = mean(A_7_ur_dim6);
E_A_8_ur_dim6 = mean(A_8_ur_dim6);
E_A_9_ur_dim6 = mean(A_9_ur_dim6);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 7 dimensionalen Unterraum
E_A_0_ur_dim7 = mean(A_0_ur_dim7);
E_A_1_ur_dim7 = mean(A_1_ur_dim7);
E_A_2_ur_dim7 = mean(A_2_ur_dim7);
E_A_3_ur_dim7 = mean(A_3_ur_dim7);
E_A_4_ur_dim7 = mean(A_4_ur_dim7);
E_A_5_ur_dim7 = mean(A_5_ur_dim7);
E_A_6_ur_dim7 = mean(A_6_ur_dim7);
E_A_7_ur_dim7 = mean(A_7_ur_dim7);
E_A_8_ur_dim7 = mean(A_8_ur_dim7);
E_A_9_ur_dim7 = mean(A_9_ur_dim7);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 8 dimensionalen Unterraum
E_A_0_ur_dim8 = mean(A_0_ur_dim8);
E_A_1_ur_dim8 = mean(A_1_ur_dim8);
E_A_2_ur_dim8 = mean(A_2_ur_dim8);
E_A_3_ur_dim8 = mean(A_3_ur_dim8);
E_A_4_ur_dim8 = mean(A_4_ur_dim8);
E_A_5_ur_dim8 = mean(A_5_ur_dim8);
E_A_6_ur_dim8 = mean(A_6_ur_dim8);
E_A_7_ur_dim8 = mean(A_7_ur_dim8);
E_A_8_ur_dim8 = mean(A_8_ur_dim8);
E_A_9_ur_dim8 = mean(A_9_ur_dim8);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 9 dimensionalen Unterraum
E_A_0_ur_dim9 = mean(A_0_ur_dim9);
E_A_1_ur_dim9 = mean(A_1_ur_dim9);
E_A_2_ur_dim9 = mean(A_2_ur_dim9);
E_A_3_ur_dim9 = mean(A_3_ur_dim9);
E_A_4_ur_dim9 = mean(A_4_ur_dim9);
E_A_5_ur_dim9 = mean(A_5_ur_dim9);
E_A_6_ur_dim9 = mean(A_6_ur_dim9);
E_A_7_ur_dim9 = mean(A_7_ur_dim9);
E_A_8_ur_dim9 = mean(A_8_ur_dim9);
E_A_9_ur_dim9 = mean(A_9_ur_dim9);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 10 dimensionalen Unterraum
E_A_0_ur_dim10 = mean(A_0_ur_dim10);
E_A_1_ur_dim10 = mean(A_1_ur_dim10);
E_A_2_ur_dim10 = mean(A_2_ur_dim10);
E_A_3_ur_dim10 = mean(A_3_ur_dim10);
E_A_4_ur_dim10 = mean(A_4_ur_dim10);
E_A_5_ur_dim10 = mean(A_5_ur_dim10);
E_A_6_ur_dim10 = mean(A_6_ur_dim10);
E_A_7_ur_dim10 = mean(A_7_ur_dim10);
E_A_8_ur_dim10 = mean(A_8_ur_dim10);
E_A_9_ur_dim10 = mean(A_9_ur_dim10);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 11 dimensionalen Unterraum
E_A_0_ur_dim11 = mean(A_0_ur_dim11);
E_A_1_ur_dim11 = mean(A_1_ur_dim11);
E_A_2_ur_dim11 = mean(A_2_ur_dim11);
E_A_3_ur_dim11 = mean(A_3_ur_dim11);
E_A_4_ur_dim11 = mean(A_4_ur_dim11);
E_A_5_ur_dim11 = mean(A_5_ur_dim11);
E_A_6_ur_dim11 = mean(A_6_ur_dim11);
E_A_7_ur_dim11 = mean(A_7_ur_dim11);
E_A_8_ur_dim11 = mean(A_8_ur_dim11);
E_A_9_ur_dim11 = mean(A_9_ur_dim11);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 1 dimensionalen Unterraum
E_A_0_ur_dim12 = mean(A_0_ur_dim12);
E_A_1_ur_dim12 = mean(A_1_ur_dim12);
E_A_2_ur_dim12 = mean(A_2_ur_dim12);
E_A_3_ur_dim12 = mean(A_3_ur_dim12);
E_A_4_ur_dim12 = mean(A_4_ur_dim12);
E_A_5_ur_dim12 = mean(A_5_ur_dim12);
E_A_6_ur_dim12 = mean(A_6_ur_dim12);
E_A_7_ur_dim12 = mean(A_7_ur_dim12);
E_A_8_ur_dim12 = mean(A_8_ur_dim12);
E_A_9_ur_dim12 = mean(A_9_ur_dim12);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 1 dimensionalen Unterraum
E_A_0_ur_dim13 = mean(A_0_ur_dim13);
E_A_1_ur_dim13 = mean(A_1_ur_dim13);
E_A_2_ur_dim13 = mean(A_2_ur_dim13);
E_A_3_ur_dim13 = mean(A_3_ur_dim13);
E_A_4_ur_dim13 = mean(A_4_ur_dim13);
E_A_5_ur_dim13 = mean(A_5_ur_dim13);
E_A_6_ur_dim13 = mean(A_6_ur_dim13);
E_A_7_ur_dim13 = mean(A_7_ur_dim13);
E_A_8_ur_dim13 = mean(A_8_ur_dim13);
E_A_9_ur_dim13 = mean(A_9_ur_dim13);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 14 dimensionalen Unterraum
E_A_0_ur_dim14 = mean(A_0_ur_dim14);
E_A_1_ur_dim14 = mean(A_1_ur_dim14);
E_A_2_ur_dim14 = mean(A_2_ur_dim14);
E_A_3_ur_dim14 = mean(A_3_ur_dim14);
E_A_4_ur_dim14 = mean(A_4_ur_dim14);
E_A_5_ur_dim14 = mean(A_5_ur_dim14);
E_A_6_ur_dim14 = mean(A_6_ur_dim14);
E_A_7_ur_dim14 = mean(A_7_ur_dim14);
E_A_8_ur_dim14 = mean(A_8_ur_dim14);
E_A_9_ur_dim14 = mean(A_9_ur_dim14);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 15 dimensionalen Unterraum
E_A_0_ur_dim15 = mean(A_0_ur_dim15);
E_A_1_ur_dim15 = mean(A_1_ur_dim15);
E_A_2_ur_dim15 = mean(A_2_ur_dim15);
E_A_3_ur_dim15 = mean(A_3_ur_dim15);
E_A_4_ur_dim15 = mean(A_4_ur_dim15);
E_A_5_ur_dim15 = mean(A_5_ur_dim15);
E_A_6_ur_dim15 = mean(A_6_ur_dim15);
E_A_7_ur_dim15 = mean(A_7_ur_dim15);
E_A_8_ur_dim15 = mean(A_8_ur_dim15);
E_A_9_ur_dim15 = mean(A_9_ur_dim15);

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9) im 16 dimensionalen Unterraum
E_A_0_ur_dim16 = mean(A_0_ur_dim16);
E_A_1_ur_dim16 = mean(A_1_ur_dim16);
E_A_2_ur_dim16 = mean(A_2_ur_dim16);
E_A_3_ur_dim16 = mean(A_3_ur_dim16);
E_A_4_ur_dim16 = mean(A_4_ur_dim16);
E_A_5_ur_dim16 = mean(A_5_ur_dim16);
E_A_6_ur_dim16 = mean(A_6_ur_dim16);
E_A_7_ur_dim16 = mean(A_7_ur_dim16);
E_A_8_ur_dim16 = mean(A_8_ur_dim16);
E_A_9_ur_dim16 = mean(A_9_ur_dim16);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 1 dimensionalen Unterraum
CVM_A_0_ur_dim1 = cov(A_0_ur_dim1);
CVM_A_1_ur_dim1 = cov(A_1_ur_dim1);
CVM_A_2_ur_dim1 = cov(A_2_ur_dim1);
CVM_A_3_ur_dim1 = cov(A_3_ur_dim1);
CVM_A_4_ur_dim1 = cov(A_4_ur_dim1);
CVM_A_5_ur_dim1 = cov(A_5_ur_dim1);
CVM_A_6_ur_dim1 = cov(A_6_ur_dim1);
CVM_A_7_ur_dim1 = cov(A_7_ur_dim1);
CVM_A_8_ur_dim1 = cov(A_8_ur_dim1);
CVM_A_9_ur_dim1 = cov(A_9_ur_dim1);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 2 dimensionalen Unterraum
CVM_A_0_ur_dim2 = cov(A_0_ur_dim2);
CVM_A_1_ur_dim2 = cov(A_1_ur_dim2);
CVM_A_2_ur_dim2 = cov(A_2_ur_dim2);
CVM_A_3_ur_dim2 = cov(A_3_ur_dim2);
CVM_A_4_ur_dim2 = cov(A_4_ur_dim2);
CVM_A_5_ur_dim2 = cov(A_5_ur_dim2);
CVM_A_6_ur_dim2 = cov(A_6_ur_dim2);
CVM_A_7_ur_dim2 = cov(A_7_ur_dim2);
CVM_A_8_ur_dim2 = cov(A_8_ur_dim2);
CVM_A_9_ur_dim2 = cov(A_9_ur_dim2);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 3 dimensionalen Unterraum
CVM_A_0_ur_dim3 = cov(A_0_ur_dim3);
CVM_A_1_ur_dim3 = cov(A_1_ur_dim3);
CVM_A_2_ur_dim3 = cov(A_2_ur_dim3);
CVM_A_3_ur_dim3 = cov(A_3_ur_dim3);
CVM_A_4_ur_dim3 = cov(A_4_ur_dim3);
CVM_A_5_ur_dim3 = cov(A_5_ur_dim3);
CVM_A_6_ur_dim3 = cov(A_6_ur_dim3);
CVM_A_7_ur_dim3 = cov(A_7_ur_dim3);
CVM_A_8_ur_dim3 = cov(A_8_ur_dim3);
CVM_A_9_ur_dim3 = cov(A_9_ur_dim3);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 4 dimensionalen Unterraum
CVM_A_0_ur_dim4 = cov(A_0_ur_dim4);
CVM_A_1_ur_dim4 = cov(A_1_ur_dim4);
CVM_A_2_ur_dim4 = cov(A_2_ur_dim4);
CVM_A_3_ur_dim4 = cov(A_3_ur_dim4);
CVM_A_4_ur_dim4 = cov(A_4_ur_dim4);
CVM_A_5_ur_dim4 = cov(A_5_ur_dim4);
CVM_A_6_ur_dim4 = cov(A_6_ur_dim4);
CVM_A_7_ur_dim4 = cov(A_7_ur_dim4);
CVM_A_8_ur_dim4 = cov(A_8_ur_dim4);
CVM_A_9_ur_dim4 = cov(A_9_ur_dim4);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 5 dimensionalen Unterraum
CVM_A_0_ur_dim5 = cov(A_0_ur_dim5);
CVM_A_1_ur_dim5 = cov(A_1_ur_dim5);
CVM_A_2_ur_dim5 = cov(A_2_ur_dim5);
CVM_A_3_ur_dim5 = cov(A_3_ur_dim5);
CVM_A_4_ur_dim5 = cov(A_4_ur_dim5);
CVM_A_5_ur_dim5 = cov(A_5_ur_dim5);
CVM_A_6_ur_dim5 = cov(A_6_ur_dim5);
CVM_A_7_ur_dim5 = cov(A_7_ur_dim5);
CVM_A_8_ur_dim5 = cov(A_8_ur_dim5);
CVM_A_9_ur_dim5 = cov(A_9_ur_dim5);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 6 dimensionalen Unterraum
CVM_A_0_ur_dim6 = cov(A_0_ur_dim6);
CVM_A_1_ur_dim6 = cov(A_1_ur_dim6);
CVM_A_2_ur_dim6 = cov(A_2_ur_dim6);
CVM_A_3_ur_dim6 = cov(A_3_ur_dim6);
CVM_A_4_ur_dim6 = cov(A_4_ur_dim6);
CVM_A_5_ur_dim6 = cov(A_5_ur_dim6);
CVM_A_6_ur_dim6 = cov(A_6_ur_dim6);
CVM_A_7_ur_dim6 = cov(A_7_ur_dim6);
CVM_A_8_ur_dim6 = cov(A_8_ur_dim6);
CVM_A_9_ur_dim6 = cov(A_9_ur_dim6);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 7 dimensionalen Unterraum
CVM_A_0_ur_dim7 = cov(A_0_ur_dim7);
CVM_A_1_ur_dim7 = cov(A_1_ur_dim7);
CVM_A_2_ur_dim7 = cov(A_2_ur_dim7);
CVM_A_3_ur_dim7 = cov(A_3_ur_dim7);
CVM_A_4_ur_dim7 = cov(A_4_ur_dim7);
CVM_A_5_ur_dim7 = cov(A_5_ur_dim7);
CVM_A_6_ur_dim7 = cov(A_6_ur_dim7);
CVM_A_7_ur_dim7 = cov(A_7_ur_dim7);
CVM_A_8_ur_dim7 = cov(A_8_ur_dim7);
CVM_A_9_ur_dim7 = cov(A_9_ur_dim7);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 8 dimensionalen Unterraum
CVM_A_0_ur_dim8 = cov(A_0_ur_dim8);
CVM_A_1_ur_dim8 = cov(A_1_ur_dim8);
CVM_A_2_ur_dim8 = cov(A_2_ur_dim8);
CVM_A_3_ur_dim8 = cov(A_3_ur_dim8);
CVM_A_4_ur_dim8 = cov(A_4_ur_dim8);
CVM_A_5_ur_dim8 = cov(A_5_ur_dim8);
CVM_A_6_ur_dim8 = cov(A_6_ur_dim8);
CVM_A_7_ur_dim8 = cov(A_7_ur_dim8);
CVM_A_8_ur_dim8 = cov(A_8_ur_dim8);
CVM_A_9_ur_dim8 = cov(A_9_ur_dim8);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 9 dimensionalen Unterraum
CVM_A_0_ur_dim9 = cov(A_0_ur_dim9);
CVM_A_1_ur_dim9 = cov(A_1_ur_dim9);
CVM_A_2_ur_dim9 = cov(A_2_ur_dim9);
CVM_A_3_ur_dim9 = cov(A_3_ur_dim9);
CVM_A_4_ur_dim9 = cov(A_4_ur_dim9);
CVM_A_5_ur_dim9 = cov(A_5_ur_dim9);
CVM_A_6_ur_dim9 = cov(A_6_ur_dim9);
CVM_A_7_ur_dim9 = cov(A_7_ur_dim9);
CVM_A_8_ur_dim9 = cov(A_8_ur_dim9);
CVM_A_9_ur_dim9 = cov(A_9_ur_dim9);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 10 dimensionalen Unterraum
CVM_A_0_ur_dim10 = cov(A_0_ur_dim10);
CVM_A_1_ur_dim10 = cov(A_1_ur_dim10);
CVM_A_2_ur_dim10 = cov(A_2_ur_dim10);
CVM_A_3_ur_dim10 = cov(A_3_ur_dim10);
CVM_A_4_ur_dim10 = cov(A_4_ur_dim10);
CVM_A_5_ur_dim10 = cov(A_5_ur_dim10);
CVM_A_6_ur_dim10 = cov(A_6_ur_dim10);
CVM_A_7_ur_dim10 = cov(A_7_ur_dim10);
CVM_A_8_ur_dim10 = cov(A_8_ur_dim10);
CVM_A_9_ur_dim10 = cov(A_9_ur_dim10);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 11 dimensionalen Unterraum
CVM_A_0_ur_dim11 = cov(A_0_ur_dim11);
CVM_A_1_ur_dim11 = cov(A_1_ur_dim11);
CVM_A_2_ur_dim11 = cov(A_2_ur_dim11);
CVM_A_3_ur_dim11 = cov(A_3_ur_dim11);
CVM_A_4_ur_dim11 = cov(A_4_ur_dim11);
CVM_A_5_ur_dim11 = cov(A_5_ur_dim11);
CVM_A_6_ur_dim11 = cov(A_6_ur_dim11);
CVM_A_7_ur_dim11 = cov(A_7_ur_dim11);
CVM_A_8_ur_dim11 = cov(A_8_ur_dim11);
CVM_A_9_ur_dim11 = cov(A_9_ur_dim11);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 12 dimensionalen Unterraum
CVM_A_0_ur_dim12 = cov(A_0_ur_dim12);
CVM_A_1_ur_dim12 = cov(A_1_ur_dim12);
CVM_A_2_ur_dim12 = cov(A_2_ur_dim12);
CVM_A_3_ur_dim12 = cov(A_3_ur_dim12);
CVM_A_4_ur_dim12 = cov(A_4_ur_dim12);
CVM_A_5_ur_dim12 = cov(A_5_ur_dim12);
CVM_A_6_ur_dim12 = cov(A_6_ur_dim12);
CVM_A_7_ur_dim12 = cov(A_7_ur_dim12);
CVM_A_8_ur_dim12 = cov(A_8_ur_dim12);
CVM_A_9_ur_dim12 = cov(A_9_ur_dim12);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 13 dimensionalen Unterraum
CVM_A_0_ur_dim13 = cov(A_0_ur_dim13);
CVM_A_1_ur_dim13 = cov(A_1_ur_dim13);
CVM_A_2_ur_dim13 = cov(A_2_ur_dim13);
CVM_A_3_ur_dim13 = cov(A_3_ur_dim13);
CVM_A_4_ur_dim13 = cov(A_4_ur_dim13);
CVM_A_5_ur_dim13 = cov(A_5_ur_dim13);
CVM_A_6_ur_dim13 = cov(A_6_ur_dim13);
CVM_A_7_ur_dim13 = cov(A_7_ur_dim13);
CVM_A_8_ur_dim13 = cov(A_8_ur_dim13);
CVM_A_9_ur_dim13 = cov(A_9_ur_dim13);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 14 dimensionalen Unterraum
CVM_A_0_ur_dim14 = cov(A_0_ur_dim14);
CVM_A_1_ur_dim14 = cov(A_1_ur_dim14);
CVM_A_2_ur_dim14 = cov(A_2_ur_dim14);
CVM_A_3_ur_dim14 = cov(A_3_ur_dim14);
CVM_A_4_ur_dim14 = cov(A_4_ur_dim14);
CVM_A_5_ur_dim14 = cov(A_5_ur_dim14);
CVM_A_6_ur_dim14 = cov(A_6_ur_dim14);
CVM_A_7_ur_dim14 = cov(A_7_ur_dim14);
CVM_A_8_ur_dim14 = cov(A_8_ur_dim14);
CVM_A_9_ur_dim14 = cov(A_9_ur_dim14);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 15 dimensionalen Unterraum
CVM_A_0_ur_dim15 = cov(A_0_ur_dim15);
CVM_A_1_ur_dim15 = cov(A_1_ur_dim15);
CVM_A_2_ur_dim15 = cov(A_2_ur_dim15);
CVM_A_3_ur_dim15 = cov(A_3_ur_dim15);
CVM_A_4_ur_dim15 = cov(A_4_ur_dim15);
CVM_A_5_ur_dim15 = cov(A_5_ur_dim15);
CVM_A_6_ur_dim15 = cov(A_6_ur_dim15);
CVM_A_7_ur_dim15 = cov(A_7_ur_dim15);
CVM_A_8_ur_dim15 = cov(A_8_ur_dim15);
CVM_A_9_ur_dim15 = cov(A_9_ur_dim15);

% Kovarianzmatrix fuer jeden Zug (0 bis 9) im 16 dimensionalen Unterraum
CVM_A_0_ur_dim16 = cov(A_0_ur_dim16);
CVM_A_1_ur_dim16 = cov(A_1_ur_dim16);
CVM_A_2_ur_dim16 = cov(A_2_ur_dim16);
CVM_A_3_ur_dim16 = cov(A_3_ur_dim16);
CVM_A_4_ur_dim16 = cov(A_4_ur_dim16);
CVM_A_5_ur_dim16 = cov(A_5_ur_dim16);
CVM_A_6_ur_dim16 = cov(A_6_ur_dim16);
CVM_A_7_ur_dim16 = cov(A_7_ur_dim16);
CVM_A_8_ur_dim16 = cov(A_8_ur_dim16);
CVM_A_9_ur_dim16 = cov(A_9_ur_dim16);


% Multivariante PDF generieren fuer jeden Zug (0 bis 9)

% multivariate PDF im 1 dimensionalen Unterraum
A_0_mvpdf_ur_dim1 = mvnpdf(A_0_ur_dim1, E_A_0_ur_dim1, CVM_A_0_ur_dim1);
A_1_mvpdf_ur_dim1 = mvnpdf(A_1_ur_dim1, E_A_1_ur_dim1, CVM_A_1_ur_dim1);
A_2_mvpdf_ur_dim1 = mvnpdf(A_2_ur_dim1, E_A_2_ur_dim1, CVM_A_2_ur_dim1);
A_3_mvpdf_ur_dim1 = mvnpdf(A_3_ur_dim1, E_A_3_ur_dim1, CVM_A_3_ur_dim1);
A_4_mvpdf_ur_dim1 = mvnpdf(A_4_ur_dim1, E_A_4_ur_dim1, CVM_A_4_ur_dim1);
A_5_mvpdf_ur_dim1 = mvnpdf(A_5_ur_dim1, E_A_5_ur_dim1, CVM_A_5_ur_dim1);
A_6_mvpdf_ur_dim1 = mvnpdf(A_6_ur_dim1, E_A_6_ur_dim1, CVM_A_6_ur_dim1);
A_7_mvpdf_ur_dim1 = mvnpdf(A_7_ur_dim1, E_A_7_ur_dim1, CVM_A_7_ur_dim1);
A_8_mvpdf_ur_dim1 = mvnpdf(A_8_ur_dim1, E_A_8_ur_dim1, CVM_A_8_ur_dim1);
A_9_mvpdf_ur_dim1 = mvnpdf(A_9_ur_dim1, E_A_9_ur_dim1, CVM_A_9_ur_dim1);

% multivariate PDF im 2 dimensionalen Unterraum
A_0_mvpdf_ur_dim2 = mvnpdf(A_0_ur_dim2, E_A_0_ur_dim2, CVM_A_0_ur_dim2);
A_1_mvpdf_ur_dim2 = mvnpdf(A_1_ur_dim2, E_A_1_ur_dim2, CVM_A_1_ur_dim2);
A_2_mvpdf_ur_dim2 = mvnpdf(A_2_ur_dim2, E_A_2_ur_dim2, CVM_A_2_ur_dim2);
A_3_mvpdf_ur_dim2 = mvnpdf(A_3_ur_dim2, E_A_3_ur_dim2, CVM_A_3_ur_dim2);
A_4_mvpdf_ur_dim2 = mvnpdf(A_4_ur_dim2, E_A_4_ur_dim2, CVM_A_4_ur_dim2);
A_5_mvpdf_ur_dim2 = mvnpdf(A_5_ur_dim2, E_A_5_ur_dim2, CVM_A_5_ur_dim2);
A_6_mvpdf_ur_dim2 = mvnpdf(A_6_ur_dim2, E_A_6_ur_dim2, CVM_A_6_ur_dim2);
A_7_mvpdf_ur_dim2 = mvnpdf(A_7_ur_dim2, E_A_7_ur_dim2, CVM_A_7_ur_dim2);
A_8_mvpdf_ur_dim2 = mvnpdf(A_8_ur_dim2, E_A_8_ur_dim2, CVM_A_8_ur_dim2);
A_9_mvpdf_ur_dim1 = mvnpdf(A_9_ur_dim2, E_A_9_ur_dim2, CVM_A_9_ur_dim2);

% multivariate PDF im 3 dimensionalen Unterraum
A_0_mvpdf_ur_dim3 = mvnpdf(A_0_ur_dim3, E_A_0_ur_dim3, CVM_A_0_ur_dim3);
A_1_mvpdf_ur_dim3 = mvnpdf(A_1_ur_dim3, E_A_1_ur_dim3, CVM_A_1_ur_dim3);
A_2_mvpdf_ur_dim3 = mvnpdf(A_2_ur_dim3, E_A_2_ur_dim3, CVM_A_2_ur_dim3);
A_3_mvpdf_ur_dim3 = mvnpdf(A_3_ur_dim3, E_A_3_ur_dim3, CVM_A_3_ur_dim3);
A_4_mvpdf_ur_dim3 = mvnpdf(A_4_ur_dim3, E_A_4_ur_dim3, CVM_A_4_ur_dim3);
A_5_mvpdf_ur_dim3 = mvnpdf(A_5_ur_dim3, E_A_5_ur_dim3, CVM_A_5_ur_dim3);
A_6_mvpdf_ur_dim3 = mvnpdf(A_6_ur_dim3, E_A_6_ur_dim3, CVM_A_6_ur_dim3);
A_7_mvpdf_ur_dim3 = mvnpdf(A_7_ur_dim3, E_A_7_ur_dim3, CVM_A_7_ur_dim3);
A_8_mvpdf_ur_dim3 = mvnpdf(A_8_ur_dim3, E_A_8_ur_dim3, CVM_A_8_ur_dim3);
A_9_mvpdf_ur_dim3 = mvnpdf(A_9_ur_dim3, E_A_9_ur_dim3, CVM_A_9_ur_dim3);

% multivariate PDF im 4 dimensionalen Unterraum
A_0_mvpdf_ur_dim4 = mvnpdf(A_0_ur_dim4, E_A_0_ur_dim4, CVM_A_0_ur_dim4);
A_1_mvpdf_ur_dim4 = mvnpdf(A_1_ur_dim4, E_A_1_ur_dim4, CVM_A_1_ur_dim4);
A_2_mvpdf_ur_dim4 = mvnpdf(A_2_ur_dim4, E_A_2_ur_dim4, CVM_A_2_ur_dim4);
A_3_mvpdf_ur_dim4 = mvnpdf(A_3_ur_dim4, E_A_3_ur_dim4, CVM_A_3_ur_dim4);
A_4_mvpdf_ur_dim4 = mvnpdf(A_4_ur_dim4, E_A_4_ur_dim4, CVM_A_4_ur_dim4);
A_5_mvpdf_ur_dim4 = mvnpdf(A_5_ur_dim4, E_A_5_ur_dim4, CVM_A_5_ur_dim4);
A_6_mvpdf_ur_dim4 = mvnpdf(A_6_ur_dim4, E_A_6_ur_dim4, CVM_A_6_ur_dim4);
A_7_mvpdf_ur_dim4 = mvnpdf(A_7_ur_dim4, E_A_7_ur_dim4, CVM_A_7_ur_dim4);
A_8_mvpdf_ur_dim4 = mvnpdf(A_8_ur_dim4, E_A_8_ur_dim4, CVM_A_8_ur_dim4);
A_9_mvpdf_ur_dim4 = mvnpdf(A_9_ur_dim4, E_A_9_ur_dim4, CVM_A_9_ur_dim4);

% multivariate PDF im 5 dimensionalen Unterraum
A_0_mvpdf_ur_dim5 = mvnpdf(A_0_ur_dim5, E_A_0_ur_dim5, CVM_A_0_ur_dim5);
A_1_mvpdf_ur_dim5 = mvnpdf(A_1_ur_dim5, E_A_1_ur_dim5, CVM_A_1_ur_dim5);
A_2_mvpdf_ur_dim5 = mvnpdf(A_2_ur_dim5, E_A_2_ur_dim5, CVM_A_2_ur_dim5);
A_3_mvpdf_ur_dim5 = mvnpdf(A_3_ur_dim5, E_A_3_ur_dim5, CVM_A_3_ur_dim5);
A_4_mvpdf_ur_dim5 = mvnpdf(A_4_ur_dim5, E_A_4_ur_dim5, CVM_A_4_ur_dim5);
A_5_mvpdf_ur_dim5 = mvnpdf(A_5_ur_dim5, E_A_5_ur_dim5, CVM_A_5_ur_dim5);
A_6_mvpdf_ur_dim5 = mvnpdf(A_6_ur_dim5, E_A_6_ur_dim5, CVM_A_6_ur_dim5);
A_7_mvpdf_ur_dim5 = mvnpdf(A_7_ur_dim5, E_A_7_ur_dim5, CVM_A_7_ur_dim5);
A_8_mvpdf_ur_dim5 = mvnpdf(A_8_ur_dim5, E_A_8_ur_dim5, CVM_A_8_ur_dim5);
A_9_mvpdf_ur_dim5 = mvnpdf(A_9_ur_dim5, E_A_9_ur_dim5, CVM_A_9_ur_dim5);

% multivariate PDF im 6 dimensionalen Unterraum
A_0_mvpdf_ur_dim6 = mvnpdf(A_0_ur_dim6, E_A_0_ur_dim6, CVM_A_0_ur_dim6);
A_1_mvpdf_ur_dim6 = mvnpdf(A_1_ur_dim6, E_A_1_ur_dim6, CVM_A_1_ur_dim6);
A_2_mvpdf_ur_dim6 = mvnpdf(A_2_ur_dim6, E_A_2_ur_dim6, CVM_A_2_ur_dim6);
A_3_mvpdf_ur_dim6 = mvnpdf(A_3_ur_dim6, E_A_3_ur_dim6, CVM_A_3_ur_dim6);
A_4_mvpdf_ur_dim6 = mvnpdf(A_4_ur_dim6, E_A_4_ur_dim6, CVM_A_4_ur_dim6);
A_5_mvpdf_ur_dim6 = mvnpdf(A_5_ur_dim6, E_A_5_ur_dim6, CVM_A_5_ur_dim6);
A_6_mvpdf_ur_dim6 = mvnpdf(A_6_ur_dim6, E_A_6_ur_dim6, CVM_A_6_ur_dim6);
A_7_mvpdf_ur_dim6 = mvnpdf(A_7_ur_dim6, E_A_7_ur_dim6, CVM_A_7_ur_dim6);
A_8_mvpdf_ur_dim6 = mvnpdf(A_8_ur_dim6, E_A_8_ur_dim6, CVM_A_8_ur_dim6);
A_9_mvpdf_ur_dim6 = mvnpdf(A_9_ur_dim6, E_A_9_ur_dim6, CVM_A_9_ur_dim6);

% multivariate PDF im 7 dimensionalen Unterraum
A_0_mvpdf_ur_dim7 = mvnpdf(A_0_ur_dim7, E_A_0_ur_dim7, CVM_A_0_ur_dim7);
A_1_mvpdf_ur_dim7 = mvnpdf(A_1_ur_dim7, E_A_1_ur_dim7, CVM_A_1_ur_dim7);
A_2_mvpdf_ur_dim7 = mvnpdf(A_2_ur_dim7, E_A_2_ur_dim7, CVM_A_2_ur_dim7);
A_3_mvpdf_ur_dim7 = mvnpdf(A_3_ur_dim7, E_A_3_ur_dim7, CVM_A_3_ur_dim7);
A_4_mvpdf_ur_dim7 = mvnpdf(A_4_ur_dim7, E_A_4_ur_dim7, CVM_A_4_ur_dim7);
A_5_mvpdf_ur_dim7 = mvnpdf(A_5_ur_dim7, E_A_5_ur_dim7, CVM_A_5_ur_dim7);
A_6_mvpdf_ur_dim7 = mvnpdf(A_6_ur_dim7, E_A_6_ur_dim7, CVM_A_6_ur_dim7);
A_7_mvpdf_ur_dim7 = mvnpdf(A_7_ur_dim7, E_A_7_ur_dim7, CVM_A_7_ur_dim7);
A_8_mvpdf_ur_dim7 = mvnpdf(A_8_ur_dim7, E_A_8_ur_dim7, CVM_A_8_ur_dim7);
A_9_mvpdf_ur_dim7 = mvnpdf(A_9_ur_dim7, E_A_9_ur_dim7, CVM_A_9_ur_dim7);

% multivariate PDF im 8 dimensionalen Unterraum
A_0_mvpdf_ur_dim8 = mvnpdf(A_0_ur_dim8, E_A_0_ur_dim8, CVM_A_0_ur_dim8);
A_1_mvpdf_ur_dim8 = mvnpdf(A_1_ur_dim8, E_A_1_ur_dim8, CVM_A_1_ur_dim8);
A_2_mvpdf_ur_dim8 = mvnpdf(A_2_ur_dim8, E_A_2_ur_dim8, CVM_A_2_ur_dim8);
A_3_mvpdf_ur_dim8 = mvnpdf(A_3_ur_dim8, E_A_3_ur_dim8, CVM_A_3_ur_dim8);
A_4_mvpdf_ur_dim8 = mvnpdf(A_4_ur_dim8, E_A_4_ur_dim8, CVM_A_4_ur_dim8);
A_5_mvpdf_ur_dim8 = mvnpdf(A_5_ur_dim8, E_A_5_ur_dim8, CVM_A_5_ur_dim8);
A_6_mvpdf_ur_dim8 = mvnpdf(A_6_ur_dim8, E_A_6_ur_dim8, CVM_A_6_ur_dim8);
A_7_mvpdf_ur_dim8 = mvnpdf(A_7_ur_dim8, E_A_7_ur_dim8, CVM_A_7_ur_dim8);
A_8_mvpdf_ur_dim8 = mvnpdf(A_8_ur_dim8, E_A_8_ur_dim8, CVM_A_8_ur_dim8);
A_9_mvpdf_ur_dim8 = mvnpdf(A_9_ur_dim8, E_A_9_ur_dim8, CVM_A_9_ur_dim8);

% multivariate PDF im 9 dimensionalen Unterraum
A_0_mvpdf_ur_dim9 = mvnpdf(A_0_ur_dim9, E_A_0_ur_dim9, CVM_A_0_ur_dim9);
A_1_mvpdf_ur_dim9 = mvnpdf(A_1_ur_dim9, E_A_1_ur_dim9, CVM_A_1_ur_dim9);
A_2_mvpdf_ur_dim9 = mvnpdf(A_2_ur_dim9, E_A_2_ur_dim9, CVM_A_2_ur_dim9);
A_3_mvpdf_ur_dim9 = mvnpdf(A_3_ur_dim9, E_A_3_ur_dim9, CVM_A_3_ur_dim9);
A_4_mvpdf_ur_dim9 = mvnpdf(A_4_ur_dim9, E_A_4_ur_dim9, CVM_A_4_ur_dim9);
A_5_mvpdf_ur_dim9 = mvnpdf(A_5_ur_dim9, E_A_5_ur_dim9, CVM_A_5_ur_dim9);
A_6_mvpdf_ur_dim9 = mvnpdf(A_6_ur_dim9, E_A_6_ur_dim9, CVM_A_6_ur_dim9);
A_7_mvpdf_ur_dim9 = mvnpdf(A_7_ur_dim9, E_A_7_ur_dim9, CVM_A_7_ur_dim9);
A_8_mvpdf_ur_dim9 = mvnpdf(A_8_ur_dim9, E_A_8_ur_dim9, CVM_A_8_ur_dim9);
A_9_mvpdf_ur_dim9 = mvnpdf(A_9_ur_dim9, E_A_9_ur_dim9, CVM_A_9_ur_dim9);

% multivariate PDF im 10 dimensionalen Unterraum
A_0_mvpdf_ur_dim10 = mvnpdf(A_0_ur_dim10, E_A_0_ur_dim10, CVM_A_0_ur_dim10);
A_1_mvpdf_ur_dim10 = mvnpdf(A_1_ur_dim10, E_A_1_ur_dim10, CVM_A_1_ur_dim10);
A_2_mvpdf_ur_dim10 = mvnpdf(A_2_ur_dim10, E_A_2_ur_dim10, CVM_A_2_ur_dim10);
A_3_mvpdf_ur_dim10 = mvnpdf(A_3_ur_dim10, E_A_3_ur_dim10, CVM_A_3_ur_dim10);
A_4_mvpdf_ur_dim10 = mvnpdf(A_4_ur_dim10, E_A_4_ur_dim10, CVM_A_4_ur_dim10);
A_5_mvpdf_ur_dim10 = mvnpdf(A_5_ur_dim10, E_A_5_ur_dim10, CVM_A_5_ur_dim10);
A_6_mvpdf_ur_dim10 = mvnpdf(A_6_ur_dim10, E_A_6_ur_dim10, CVM_A_6_ur_dim10);
A_7_mvpdf_ur_dim10 = mvnpdf(A_7_ur_dim10, E_A_7_ur_dim10, CVM_A_7_ur_dim10);
A_8_mvpdf_ur_dim10 = mvnpdf(A_8_ur_dim10, E_A_8_ur_dim10, CVM_A_8_ur_dim10);
A_9_mvpdf_ur_dim10 = mvnpdf(A_9_ur_dim10, E_A_9_ur_dim10, CVM_A_9_ur_dim10);

% multivariate PDF im 11 dimensionalen Unterraum
A_0_mvpdf_ur_dim11 = mvnpdf(A_0_ur_dim11, E_A_0_ur_dim11, CVM_A_0_ur_dim11);
A_1_mvpdf_ur_dim11 = mvnpdf(A_1_ur_dim11, E_A_1_ur_dim11, CVM_A_1_ur_dim11);
A_2_mvpdf_ur_dim11 = mvnpdf(A_2_ur_dim11, E_A_2_ur_dim11, CVM_A_2_ur_dim11);
A_3_mvpdf_ur_dim11 = mvnpdf(A_3_ur_dim11, E_A_3_ur_dim11, CVM_A_3_ur_dim11);
A_4_mvpdf_ur_dim11 = mvnpdf(A_4_ur_dim11, E_A_4_ur_dim11, CVM_A_4_ur_dim11);
A_5_mvpdf_ur_dim11 = mvnpdf(A_5_ur_dim11, E_A_5_ur_dim11, CVM_A_5_ur_dim11);
A_6_mvpdf_ur_dim11 = mvnpdf(A_6_ur_dim11, E_A_6_ur_dim11, CVM_A_6_ur_dim11);
A_7_mvpdf_ur_dim11 = mvnpdf(A_7_ur_dim11, E_A_7_ur_dim11, CVM_A_7_ur_dim11);
A_8_mvpdf_ur_dim11 = mvnpdf(A_8_ur_dim11, E_A_8_ur_dim11, CVM_A_8_ur_dim11);
A_9_mvpdf_ur_dim11 = mvnpdf(A_9_ur_dim11, E_A_9_ur_dim11, CVM_A_9_ur_dim11);

% multivariate PDF im 12 dimensionalen Unterraum
A_0_mvpdf_ur_dim12 = mvnpdf(A_0_ur_dim12, E_A_0_ur_dim12, CVM_A_0_ur_dim12);
A_1_mvpdf_ur_dim12 = mvnpdf(A_1_ur_dim12, E_A_1_ur_dim12, CVM_A_1_ur_dim12);
A_2_mvpdf_ur_dim12 = mvnpdf(A_2_ur_dim12, E_A_2_ur_dim12, CVM_A_2_ur_dim12);
A_3_mvpdf_ur_dim12 = mvnpdf(A_3_ur_dim12, E_A_3_ur_dim12, CVM_A_3_ur_dim12);
A_4_mvpdf_ur_dim12 = mvnpdf(A_4_ur_dim12, E_A_4_ur_dim12, CVM_A_4_ur_dim12);
A_5_mvpdf_ur_dim12 = mvnpdf(A_5_ur_dim12, E_A_5_ur_dim12, CVM_A_5_ur_dim12);
A_6_mvpdf_ur_dim12 = mvnpdf(A_6_ur_dim12, E_A_6_ur_dim12, CVM_A_6_ur_dim12);
A_7_mvpdf_ur_dim12 = mvnpdf(A_7_ur_dim12, E_A_7_ur_dim12, CVM_A_7_ur_dim12);
A_8_mvpdf_ur_dim12 = mvnpdf(A_8_ur_dim12, E_A_8_ur_dim12, CVM_A_8_ur_dim12);
A_9_mvpdf_ur_dim12 = mvnpdf(A_9_ur_dim12, E_A_9_ur_dim12, CVM_A_9_ur_dim12);

% multivariate PDF im 13 dimensionalen Unterraum
A_0_mvpdf_ur_dim13 = mvnpdf(A_0_ur_dim13, E_A_0_ur_dim13, CVM_A_0_ur_dim13);
A_1_mvpdf_ur_dim13 = mvnpdf(A_1_ur_dim13, E_A_1_ur_dim13, CVM_A_1_ur_dim13);
A_2_mvpdf_ur_dim13 = mvnpdf(A_2_ur_dim13, E_A_2_ur_dim13, CVM_A_2_ur_dim13);
A_3_mvpdf_ur_dim13 = mvnpdf(A_3_ur_dim13, E_A_3_ur_dim13, CVM_A_3_ur_dim13);
A_4_mvpdf_ur_dim13 = mvnpdf(A_4_ur_dim13, E_A_4_ur_dim13, CVM_A_4_ur_dim13);
A_5_mvpdf_ur_dim13 = mvnpdf(A_5_ur_dim13, E_A_5_ur_dim13, CVM_A_5_ur_dim13);
A_6_mvpdf_ur_dim13 = mvnpdf(A_6_ur_dim13, E_A_6_ur_dim13, CVM_A_6_ur_dim13);
A_7_mvpdf_ur_dim13 = mvnpdf(A_7_ur_dim13, E_A_7_ur_dim13, CVM_A_7_ur_dim13);
A_8_mvpdf_ur_dim13 = mvnpdf(A_8_ur_dim13, E_A_8_ur_dim13, CVM_A_8_ur_dim13);
A_9_mvpdf_ur_dim13 = mvnpdf(A_9_ur_dim13, E_A_9_ur_dim13, CVM_A_9_ur_dim13);

% multivariate PDF im 14 dimensionalen Unterraum
A_0_mvpdf_ur_dim14 = mvnpdf(A_0_ur_dim14, E_A_0_ur_dim14, CVM_A_0_ur_dim14);
A_1_mvpdf_ur_dim14 = mvnpdf(A_1_ur_dim14, E_A_1_ur_dim14, CVM_A_1_ur_dim14);
A_2_mvpdf_ur_dim14 = mvnpdf(A_2_ur_dim14, E_A_2_ur_dim14, CVM_A_2_ur_dim14);
A_3_mvpdf_ur_dim14 = mvnpdf(A_3_ur_dim14, E_A_3_ur_dim14, CVM_A_3_ur_dim14);
A_4_mvpdf_ur_dim14 = mvnpdf(A_4_ur_dim14, E_A_4_ur_dim14, CVM_A_4_ur_dim14);
A_5_mvpdf_ur_dim14 = mvnpdf(A_5_ur_dim14, E_A_5_ur_dim14, CVM_A_5_ur_dim14);
A_6_mvpdf_ur_dim14 = mvnpdf(A_6_ur_dim14, E_A_6_ur_dim14, CVM_A_6_ur_dim14);
A_7_mvpdf_ur_dim14 = mvnpdf(A_7_ur_dim14, E_A_7_ur_dim14, CVM_A_7_ur_dim14);
A_8_mvpdf_ur_dim14 = mvnpdf(A_8_ur_dim14, E_A_8_ur_dim14, CVM_A_8_ur_dim14);
A_9_mvpdf_ur_dim14 = mvnpdf(A_9_ur_dim14, E_A_9_ur_dim14, CVM_A_9_ur_dim14);

% multivariate PDF im 15 dimensionalen Unterraum
A_0_mvpdf_ur_dim15 = mvnpdf(A_0_ur_dim15, E_A_0_ur_dim15, CVM_A_0_ur_dim15);
A_1_mvpdf_ur_dim15 = mvnpdf(A_1_ur_dim15, E_A_1_ur_dim15, CVM_A_1_ur_dim15);
A_2_mvpdf_ur_dim15 = mvnpdf(A_2_ur_dim15, E_A_2_ur_dim15, CVM_A_2_ur_dim15);
A_3_mvpdf_ur_dim15 = mvnpdf(A_3_ur_dim15, E_A_3_ur_dim15, CVM_A_3_ur_dim15);
A_4_mvpdf_ur_dim15 = mvnpdf(A_4_ur_dim15, E_A_4_ur_dim15, CVM_A_4_ur_dim15);
A_5_mvpdf_ur_dim15 = mvnpdf(A_5_ur_dim15, E_A_5_ur_dim15, CVM_A_5_ur_dim15);
A_6_mvpdf_ur_dim15 = mvnpdf(A_6_ur_dim15, E_A_6_ur_dim15, CVM_A_6_ur_dim15);
A_7_mvpdf_ur_dim15 = mvnpdf(A_7_ur_dim15, E_A_7_ur_dim15, CVM_A_7_ur_dim15);
A_8_mvpdf_ur_dim15 = mvnpdf(A_8_ur_dim15, E_A_8_ur_dim15, CVM_A_8_ur_dim15);
A_9_mvpdf_ur_dim15 = mvnpdf(A_9_ur_dim15, E_A_9_ur_dim15, CVM_A_9_ur_dim15);

% multivariate PDF im 16 dimensionalen Unterraum
A_0_mvpdf_ur_dim16 = mvnpdf(A_0_ur_dim16, E_A_0_ur_dim16, CVM_A_0_ur_dim16);
A_1_mvpdf_ur_dim16 = mvnpdf(A_1_ur_dim16, E_A_1_ur_dim16, CVM_A_1_ur_dim16);
A_2_mvpdf_ur_dim16 = mvnpdf(A_2_ur_dim16, E_A_2_ur_dim16, CVM_A_2_ur_dim16);
A_3_mvpdf_ur_dim16 = mvnpdf(A_3_ur_dim16, E_A_3_ur_dim16, CVM_A_3_ur_dim16);
A_4_mvpdf_ur_dim16 = mvnpdf(A_4_ur_dim16, E_A_4_ur_dim16, CVM_A_4_ur_dim16);
A_5_mvpdf_ur_dim16 = mvnpdf(A_5_ur_dim16, E_A_5_ur_dim16, CVM_A_5_ur_dim16);
A_6_mvpdf_ur_dim16 = mvnpdf(A_6_ur_dim16, E_A_6_ur_dim16, CVM_A_6_ur_dim16);
A_7_mvpdf_ur_dim16 = mvnpdf(A_7_ur_dim16, E_A_7_ur_dim16, CVM_A_7_ur_dim16);
A_8_mvpdf_ur_dim16 = mvnpdf(A_8_ur_dim16, E_A_8_ur_dim16, CVM_A_8_ur_dim16);
A_9_mvpdf_ur_dim16 = mvnpdf(A_9_ur_dim16, E_A_9_ur_dim16, CVM_A_9_ur_dim16);

% Aposteriori - Wahrscheinlichkeit im 1 dimensionalen Unterraum
A_0_aposteriori_ur_dim1 = A_0_mvpdf_ur_dim1 * A_x_apriori;
A_1_aposteriori_ur_dim1 = A_1_mvpdf_ur_dim1 * A_x_apriori;
A_2_aposteriori_ur_dim1 = A_2_mvpdf_ur_dim1 * A_x_apriori;
A_3_aposteriori_ur_dim1 = A_3_mvpdf_ur_dim1 * A_x_apriori;
A_4_aposteriori_ur_dim1 = A_4_mvpdf_ur_dim1 * A_x_apriori;
A_5_aposteriori_ur_dim1 = A_5_mvpdf_ur_dim1 * A_x_apriori;
A_6_aposteriori_ur_dim1 = A_6_mvpdf_ur_dim1 * A_x_apriori;
A_7_aposteriori_ur_dim1 = A_7_mvpdf_ur_dim1 * A_x_apriori;
A_8_aposteriori_ur_dim1 = A_8_mvpdf_ur_dim1 * A_x_apriori;
A_9_aposteriori_ur_dim1 = A_9_mvpdf_ur_dim1 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 2 dimensionalen Unterraum
A_0_aposteriori_ur_dim2 = A_0_mvpdf_ur_dim2 * A_x_apriori;
A_1_aposteriori_ur_dim2 = A_1_mvpdf_ur_dim2 * A_x_apriori;
A_2_aposteriori_ur_dim2 = A_2_mvpdf_ur_dim2 * A_x_apriori;
A_3_aposteriori_ur_dim2 = A_3_mvpdf_ur_dim2 * A_x_apriori;
A_4_aposteriori_ur_dim2 = A_4_mvpdf_ur_dim2 * A_x_apriori;
A_5_aposteriori_ur_dim2 = A_5_mvpdf_ur_dim2 * A_x_apriori;
A_6_aposteriori_ur_dim2 = A_6_mvpdf_ur_dim2 * A_x_apriori;
A_7_aposteriori_ur_dim2 = A_7_mvpdf_ur_dim2 * A_x_apriori;
A_8_aposteriori_ur_dim2 = A_8_mvpdf_ur_dim2 * A_x_apriori;
A_9_aposteriori_ur_dim2 = A_9_mvpdf_ur_dim2 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 3 dimensionalen Unterraum
A_0_aposteriori_ur_dim3 = A_0_mvpdf_ur_dim3 * A_x_apriori;
A_1_aposteriori_ur_dim3 = A_1_mvpdf_ur_dim3 * A_x_apriori;
A_2_aposteriori_ur_dim3 = A_2_mvpdf_ur_dim3 * A_x_apriori;
A_3_aposteriori_ur_dim3 = A_3_mvpdf_ur_dim3 * A_x_apriori;
A_4_aposteriori_ur_dim3 = A_4_mvpdf_ur_dim3 * A_x_apriori;
A_5_aposteriori_ur_dim3 = A_5_mvpdf_ur_dim3 * A_x_apriori;
A_6_aposteriori_ur_dim3 = A_6_mvpdf_ur_dim3 * A_x_apriori;
A_7_aposteriori_ur_dim3 = A_7_mvpdf_ur_dim3 * A_x_apriori;
A_8_aposteriori_ur_dim3 = A_8_mvpdf_ur_dim3 * A_x_apriori;
A_9_aposteriori_ur_dim3 = A_9_mvpdf_ur_dim3 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 4 dimensionalen Unterraum
A_0_aposteriori_ur_dim4 = A_0_mvpdf_ur_dim4 * A_x_apriori;
A_1_aposteriori_ur_dim4 = A_1_mvpdf_ur_dim4 * A_x_apriori;
A_2_aposteriori_ur_dim4 = A_2_mvpdf_ur_dim4 * A_x_apriori;
A_3_aposteriori_ur_dim4 = A_3_mvpdf_ur_dim4 * A_x_apriori;
A_4_aposteriori_ur_dim4 = A_4_mvpdf_ur_dim4 * A_x_apriori;
A_5_aposteriori_ur_dim4 = A_5_mvpdf_ur_dim4 * A_x_apriori;
A_6_aposteriori_ur_dim4 = A_6_mvpdf_ur_dim4 * A_x_apriori;
A_7_aposteriori_ur_dim4 = A_7_mvpdf_ur_dim4 * A_x_apriori;
A_8_aposteriori_ur_dim4 = A_8_mvpdf_ur_dim4 * A_x_apriori;
A_9_aposteriori_ur_dim4 = A_9_mvpdf_ur_dim4 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 5 dimensionalen Unterraum
A_0_aposteriori_ur_dim5 = A_0_mvpdf_ur_dim5 * A_x_apriori;
A_1_aposteriori_ur_dim5 = A_1_mvpdf_ur_dim5 * A_x_apriori;
A_2_aposteriori_ur_dim5 = A_2_mvpdf_ur_dim5 * A_x_apriori;
A_3_aposteriori_ur_dim5 = A_3_mvpdf_ur_dim5 * A_x_apriori;
A_4_aposteriori_ur_dim5 = A_4_mvpdf_ur_dim5 * A_x_apriori;
A_5_aposteriori_ur_dim5 = A_5_mvpdf_ur_dim5 * A_x_apriori;
A_6_aposteriori_ur_dim5 = A_6_mvpdf_ur_dim5 * A_x_apriori;
A_7_aposteriori_ur_dim5 = A_7_mvpdf_ur_dim5 * A_x_apriori;
A_8_aposteriori_ur_dim5 = A_8_mvpdf_ur_dim5 * A_x_apriori;
A_9_aposteriori_ur_dim5 = A_9_mvpdf_ur_dim5 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 6 dimensionalen Unterraum
A_0_aposteriori_ur_dim6 = A_0_mvpdf_ur_dim6 * A_x_apriori;
A_1_aposteriori_ur_dim6 = A_1_mvpdf_ur_dim6 * A_x_apriori;
A_2_aposteriori_ur_dim6 = A_2_mvpdf_ur_dim6 * A_x_apriori;
A_3_aposteriori_ur_dim6 = A_3_mvpdf_ur_dim6 * A_x_apriori;
A_4_aposteriori_ur_dim6 = A_4_mvpdf_ur_dim6 * A_x_apriori;
A_5_aposteriori_ur_dim6 = A_5_mvpdf_ur_dim6 * A_x_apriori;
A_6_aposteriori_ur_dim6 = A_6_mvpdf_ur_dim6 * A_x_apriori;
A_7_aposteriori_ur_dim6 = A_7_mvpdf_ur_dim6 * A_x_apriori;
A_8_aposteriori_ur_dim6 = A_8_mvpdf_ur_dim6 * A_x_apriori;
A_9_aposteriori_ur_dim6 = A_9_mvpdf_ur_dim6 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 7 dimensionalen Unterraum
A_0_aposteriori_ur_dim7 = A_0_mvpdf_ur_dim7 * A_x_apriori;
A_1_aposteriori_ur_dim7 = A_1_mvpdf_ur_dim7 * A_x_apriori;
A_2_aposteriori_ur_dim7 = A_2_mvpdf_ur_dim7 * A_x_apriori;
A_3_aposteriori_ur_dim7 = A_3_mvpdf_ur_dim7 * A_x_apriori;
A_4_aposteriori_ur_dim7 = A_4_mvpdf_ur_dim7 * A_x_apriori;
A_5_aposteriori_ur_dim7 = A_5_mvpdf_ur_dim7 * A_x_apriori;
A_6_aposteriori_ur_dim7 = A_6_mvpdf_ur_dim7 * A_x_apriori;
A_7_aposteriori_ur_dim7 = A_7_mvpdf_ur_dim7 * A_x_apriori;
A_8_aposteriori_ur_dim7 = A_8_mvpdf_ur_dim7 * A_x_apriori;
A_9_aposteriori_ur_dim7 = A_9_mvpdf_ur_dim7 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 8 dimensionalen Unterraum
A_0_aposteriori_ur_dim8 = A_0_mvpdf_ur_dim8 * A_x_apriori;
A_1_aposteriori_ur_dim8 = A_1_mvpdf_ur_dim8 * A_x_apriori;
A_2_aposteriori_ur_dim8 = A_2_mvpdf_ur_dim8 * A_x_apriori;
A_3_aposteriori_ur_dim8 = A_3_mvpdf_ur_dim8 * A_x_apriori;
A_4_aposteriori_ur_dim8 = A_4_mvpdf_ur_dim8 * A_x_apriori;
A_5_aposteriori_ur_dim8 = A_5_mvpdf_ur_dim8 * A_x_apriori;
A_6_aposteriori_ur_dim8 = A_6_mvpdf_ur_dim8 * A_x_apriori;
A_7_aposteriori_ur_dim8 = A_7_mvpdf_ur_dim8 * A_x_apriori;
A_8_aposteriori_ur_dim8 = A_8_mvpdf_ur_dim8 * A_x_apriori;
A_9_aposteriori_ur_dim8 = A_9_mvpdf_ur_dim8 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 9 dimensionalen Unterraum
A_0_aposteriori_ur_dim9 = A_0_mvpdf_ur_dim9 * A_x_apriori;
A_1_aposteriori_ur_dim9 = A_1_mvpdf_ur_dim9 * A_x_apriori;
A_2_aposteriori_ur_dim9 = A_2_mvpdf_ur_dim9 * A_x_apriori;
A_3_aposteriori_ur_dim9 = A_3_mvpdf_ur_dim9 * A_x_apriori;
A_4_aposteriori_ur_dim9 = A_4_mvpdf_ur_dim9 * A_x_apriori;
A_5_aposteriori_ur_dim9 = A_5_mvpdf_ur_dim9 * A_x_apriori;
A_6_aposteriori_ur_dim9 = A_6_mvpdf_ur_dim9 * A_x_apriori;
A_7_aposteriori_ur_dim9 = A_7_mvpdf_ur_dim9 * A_x_apriori;
A_8_aposteriori_ur_dim9 = A_8_mvpdf_ur_dim9 * A_x_apriori;
A_9_aposteriori_ur_dim9 = A_9_mvpdf_ur_dim9 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 10 dimensionalen Unterraum
A_0_aposteriori_ur_dim10 = A_0_mvpdf_ur_dim10 * A_x_apriori;
A_1_aposteriori_ur_dim10 = A_1_mvpdf_ur_dim10 * A_x_apriori;
A_2_aposteriori_ur_dim10 = A_2_mvpdf_ur_dim10 * A_x_apriori;
A_3_aposteriori_ur_dim10 = A_3_mvpdf_ur_dim10 * A_x_apriori;
A_4_aposteriori_ur_dim10 = A_4_mvpdf_ur_dim10 * A_x_apriori;
A_5_aposteriori_ur_dim10 = A_5_mvpdf_ur_dim10 * A_x_apriori;
A_6_aposteriori_ur_dim10 = A_6_mvpdf_ur_dim10 * A_x_apriori;
A_7_aposteriori_ur_dim10 = A_7_mvpdf_ur_dim10 * A_x_apriori;
A_8_aposteriori_ur_dim10 = A_8_mvpdf_ur_dim10 * A_x_apriori;
A_9_aposteriori_ur_dim10 = A_9_mvpdf_ur_dim10 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 11 dimensionalen Unterraum
A_0_aposteriori_ur_dim11 = A_0_mvpdf_ur_dim11 * A_x_apriori;
A_1_aposteriori_ur_dim11 = A_1_mvpdf_ur_dim11 * A_x_apriori;
A_2_aposteriori_ur_dim11 = A_2_mvpdf_ur_dim11 * A_x_apriori;
A_3_aposteriori_ur_dim11 = A_3_mvpdf_ur_dim11 * A_x_apriori;
A_4_aposteriori_ur_dim11 = A_4_mvpdf_ur_dim11 * A_x_apriori;
A_5_aposteriori_ur_dim11 = A_5_mvpdf_ur_dim11 * A_x_apriori;
A_6_aposteriori_ur_dim11 = A_6_mvpdf_ur_dim11 * A_x_apriori;
A_7_aposteriori_ur_dim11 = A_7_mvpdf_ur_dim11 * A_x_apriori;
A_8_aposteriori_ur_dim11 = A_8_mvpdf_ur_dim11 * A_x_apriori;
A_9_aposteriori_ur_dim11 = A_9_mvpdf_ur_dim11 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 12 dimensionalen Unterraum
A_0_aposteriori_ur_dim12 = A_0_mvpdf_ur_dim12 * A_x_apriori;
A_1_aposteriori_ur_dim12 = A_1_mvpdf_ur_dim12 * A_x_apriori;
A_2_aposteriori_ur_dim12 = A_2_mvpdf_ur_dim12 * A_x_apriori;
A_3_aposteriori_ur_dim12 = A_3_mvpdf_ur_dim12 * A_x_apriori;
A_4_aposteriori_ur_dim12 = A_4_mvpdf_ur_dim12 * A_x_apriori;
A_5_aposteriori_ur_dim12 = A_5_mvpdf_ur_dim12 * A_x_apriori;
A_6_aposteriori_ur_dim12 = A_6_mvpdf_ur_dim12 * A_x_apriori;
A_7_aposteriori_ur_dim12 = A_7_mvpdf_ur_dim12 * A_x_apriori;
A_8_aposteriori_ur_dim12 = A_8_mvpdf_ur_dim12 * A_x_apriori;
A_9_aposteriori_ur_dim12 = A_9_mvpdf_ur_dim12 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 13 dimensionalen Unterraum
A_0_aposteriori_ur_dim13 = A_0_mvpdf_ur_dim13 * A_x_apriori;
A_1_aposteriori_ur_dim13 = A_1_mvpdf_ur_dim13 * A_x_apriori;
A_2_aposteriori_ur_dim13 = A_2_mvpdf_ur_dim13 * A_x_apriori;
A_3_aposteriori_ur_dim13 = A_3_mvpdf_ur_dim13 * A_x_apriori;
A_4_aposteriori_ur_dim13 = A_4_mvpdf_ur_dim13 * A_x_apriori;
A_5_aposteriori_ur_dim13 = A_5_mvpdf_ur_dim13 * A_x_apriori;
A_6_aposteriori_ur_dim13 = A_6_mvpdf_ur_dim13 * A_x_apriori;
A_7_aposteriori_ur_dim13 = A_7_mvpdf_ur_dim13 * A_x_apriori;
A_8_aposteriori_ur_dim13 = A_8_mvpdf_ur_dim13 * A_x_apriori;
A_9_aposteriori_ur_dim13 = A_9_mvpdf_ur_dim13 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 14 dimensionalen Unterraum
A_0_aposteriori_ur_dim14 = A_0_mvpdf_ur_dim14 * A_x_apriori;
A_1_aposteriori_ur_dim14 = A_1_mvpdf_ur_dim14 * A_x_apriori;
A_2_aposteriori_ur_dim14 = A_2_mvpdf_ur_dim14 * A_x_apriori;
A_3_aposteriori_ur_dim14 = A_3_mvpdf_ur_dim14 * A_x_apriori;
A_4_aposteriori_ur_dim14 = A_4_mvpdf_ur_dim14 * A_x_apriori;
A_5_aposteriori_ur_dim14 = A_5_mvpdf_ur_dim14 * A_x_apriori;
A_6_aposteriori_ur_dim14 = A_6_mvpdf_ur_dim14 * A_x_apriori;
A_7_aposteriori_ur_dim14 = A_7_mvpdf_ur_dim14 * A_x_apriori;
A_8_aposteriori_ur_dim14 = A_8_mvpdf_ur_dim14 * A_x_apriori;
A_9_aposteriori_ur_dim14 = A_9_mvpdf_ur_dim14 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 15 dimensionalen Unterraum
A_0_aposteriori_ur_dim15 = A_0_mvpdf_ur_dim15 * A_x_apriori;
A_1_aposteriori_ur_dim15 = A_1_mvpdf_ur_dim15 * A_x_apriori;
A_2_aposteriori_ur_dim15 = A_2_mvpdf_ur_dim15 * A_x_apriori;
A_3_aposteriori_ur_dim15 = A_3_mvpdf_ur_dim15 * A_x_apriori;
A_4_aposteriori_ur_dim15 = A_4_mvpdf_ur_dim15 * A_x_apriori;
A_5_aposteriori_ur_dim15 = A_5_mvpdf_ur_dim15 * A_x_apriori;
A_6_aposteriori_ur_dim15 = A_6_mvpdf_ur_dim15 * A_x_apriori;
A_7_aposteriori_ur_dim15 = A_7_mvpdf_ur_dim15 * A_x_apriori;
A_8_aposteriori_ur_dim15 = A_8_mvpdf_ur_dim15 * A_x_apriori;
A_9_aposteriori_ur_dim15 = A_9_mvpdf_ur_dim15 * A_x_apriori;

% Aposteriori - Wahrscheinlichkeit im 16 dimensionalen Unterraum
A_0_aposteriori_ur_dim16 = A_0_mvpdf_ur_dim16 * A_x_apriori;
A_1_aposteriori_ur_dim16 = A_1_mvpdf_ur_dim16 * A_x_apriori;
A_2_aposteriori_ur_dim16 = A_2_mvpdf_ur_dim16 * A_x_apriori;
A_3_aposteriori_ur_dim16 = A_3_mvpdf_ur_dim16 * A_x_apriori;
A_4_aposteriori_ur_dim16 = A_4_mvpdf_ur_dim16 * A_x_apriori;
A_5_aposteriori_ur_dim16 = A_5_mvpdf_ur_dim16 * A_x_apriori;
A_6_aposteriori_ur_dim16 = A_6_mvpdf_ur_dim16 * A_x_apriori;
A_7_aposteriori_ur_dim16 = A_7_mvpdf_ur_dim16 * A_x_apriori;
A_8_aposteriori_ur_dim16 = A_8_mvpdf_ur_dim16 * A_x_apriori;
A_9_aposteriori_ur_dim16 = A_9_mvpdf_ur_dim16 * A_x_apriori;

% HIER KLASSIFIZIEREN %

%%%%%%%%%%%%%%%%%%%%%%%%%%%  Aufgabe 3 (3 Punkte)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A 3.1: k-means auf die Daten clusters.txt anwenden,
%        k-means soll selbst implementiert werden!

k = 3;
numIterations = 5;

mean1 = C(1,:); % mean1, selected randomly
mean2 = C(2,:); % mean2, selected randomly
mean3 = C(3,:); % mean3, selected randomly
mean1_elems = []; % elements belonging to mean1
mean2_elems = []; % elements belonging to mean2
mean3_elems = []; % elements belonging to mean3
plotArray = [];

for iter=1:numIterations
    mean1_elems = [];
    mean2_elems = [];
    mean3_elems = [];
    for elem=1:size(C,1) % iterate over all elements
        dist = sqrt(abs(C(elem,1) - mean1(:,1))^2  + abs(C(elem,2) - mean1(:,2))^2);
        closest = mean1;
        dist2 = sqrt(abs(C(elem,1) - mean2(:,1))^2  + abs(C(elem,2) - mean2(:,2))^2);
        if dist > dist2
            closest = mean2;
            dist = dist2;
        end
        dist3 = sqrt(abs(C(elem,1) - mean3(:,1))^2  + abs(C(elem,2) - mean3(:,2))^2);
        if dist > dist3
            closest = mean3;
            dist = dist3;
        end
        if closest == mean1
            mean1_elems = vertcat(mean1_elems, C(elem, :));
        elseif closest == mean2
            mean2_elems = vertcat(mean2_elems, C(elem, :));
        else
            mean3_elems = vertcat(mean3_elems, C(elem, :));
        end
    end
    mean1_elems;
    mean2_elems;
    mean3_elems;
    mean1 = [mean(mean1_elems(:,1)), mean(mean1_elems(:,2))];
    mean2 = [mean(mean2_elems(:,1)), mean(mean2_elems(:,2))];
    mean3 = [mean(mean3_elems(:,1)), mean(mean3_elems(:,2))];
    
    % Visualisierung der Clusterzentren
    plotOfIteration = 5; % which iteration do we want to see a plot for?
    if iter == plotOfIteration
        % x = min(mean1_elems):max(mean1_elems)
        mean1_elems_x = mean1_elems(:,1); % x coordinates of all elements belonging to mean1
        mean1_elems_y = mean1_elems(:,2); % y coordinates of all elements belonging to mean1
        mean2_elems_x = mean2_elems(:,1);
        mean2_elems_y = mean2_elems(:,2);
        mean3_elems_x = mean3_elems(:,1);
        mean3_elems_y = mean3_elems(:,2);
        scatter(mean1_elems_x, mean1_elems_y, 40, [1 0 0])
        hold on
        scatter(mean1(:,1), mean1(:,2), 60, [.3 0 0], 'filled')
        hold on
        scatter(mean2_elems_x, mean2_elems_y, 40, [0 1 0])
        hold on
        scatter(mean2(:,1), mean2(:,2), 60, [0 .3 0], 'filled')
        hold on
        scatter(mean3_elems_x, mean3_elems_y, 40, [0 0 1])
        hold on
        scatter(mean3(:,1), mean3(:,2), 60, [0 0 .3], 'filled')
    end
end
