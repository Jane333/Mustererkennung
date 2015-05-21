% Daten laden (Trainings- und Testdaten)
A = load('pendigits-training.txt');
B = load('pendigits-testing.txt');

%Dimensionen der Trainingsdaten
A_n   = size(A,2);
A_m   = size(A,1);

% Dimensionen der Testdaten
B_n   = size(B,2);
B_m   = size(B,1);

% Daten ohne die Zugliniennummer (Trainings- und Testdaten)
A_nl = A(:,1:A_n -1);
B_nl = B(:,1:B_n -1);

% Trainingsdaten aufgeteilt nach Zugliniennummer
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

% Trainingsdaten aufgeteilt nach Zugliniennummer ohne Zugliniennummer (nl =
% no line)
A_0_nl = A_0(:,1:A_n -1);
A_1_nl = A_1(:,1:A_n -1);
A_2_nl = A_2(:,1:A_n -1);
A_3_nl = A_3(:,1:A_n -1);
A_4_nl = A_4(:,1:A_n -1);
A_5_nl = A_5(:,1:A_n -1);
A_6_nl = A_6(:,1:A_n -1);
A_7_nl = A_7(:,1:A_n -1);
A_8_nl = A_8(:,1:A_n -1);
A_9_nl = A_9(:,1:A_n -1);

%%%%% Aufgabe 1 (3 Punkte) %%%%%

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

% A-Priori-Wahrscheinlichkeit fuer jeden Zug (0 bis 9)
A_x_apriori = 1 / length(unique(A(:,A_n)));

% A-Posteriori-Wahrscheinlichkeit fuer jeden Zug (0 bis 9)
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify = [];
for index = 1:size(B,1)
    trainData = B(index,1:B_n -1);
    
    % multivariate PDF für Testdatensatz (fuer jede Zuglinie)
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
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([norm(A_0_aposteriori_predict),norm(A_1_aposteriori_predict),norm(A_2_aposteriori_predict),norm(A_3_aposteriori_predict),norm(A_4_aposteriori_predict),norm(A_5_aposteriori_predict),norm(A_6_aposteriori_predict),norm(A_7_aposteriori_predict),norm(A_8_aposteriori_predict),norm(A_9_aposteriori_predict)]);
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groesste?)
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
confusionmatrix = confusionmat(knownClass, predictedClass)

% Klassifikationsguete
M_m = size(M_classify, 1);
corret_predicted = 0;
for index = 1:M_m
    if M_classify(index, B_n) == M_classify(index, B_n +1)
        corret_predicted = corret_predicted + 1;
    end
end
classification_quality = corret_predicted / M_m

%%%%% Aufgabe 2 %%%%%

%%% A 2.1: Erste Hauptkomponente der Trainingsdaten

% Kovarianzmatrix
CVM_A = cov(A_nl); % zentriert durch cov()
CVM_B = cov(B_nl); % zentriert durch cov()

% Eigenvektoren (VB) und Eigenwerte (DB) der Kovarianzmatrix (balanciert)
[VB,DB] = eig(CVM_A);
EigVec_CVM_A = VB; % Eigenvektoren von CVM_A
EigVal_CVM_A = DB; % Diagonalmatrix der Eigenwerte zu CVM_A

[VB,DB] = eig(CVM_B);
EigVec_CVM_B = VB; % Eigenvektoren von CVM_B
EigVal_CVM_B = DB; % Diagonalmatrix der Eigenwerte zu CVM_B      

% Ausgabe der Hauptkomponente
first_principal_component  = EigVec_CVM_A(:,1)
biggest_principalcomponent = EigVec_CVM_A(:,size(EigVec_CVM_A,2));


%%% A 2.2: PCA und Klassiifikation der Testdaten

%%%%% PCA: 1-dimmensional %%%%%
biggestEVec = size(EigVec_CVM_A,2);

% Unterraum erzeugen
pca_ur_1dim  = EigVec_CVM_A(:,biggestEVec); % groeßter (= letzter) Eigenvektor

% Abbildung der Trainingsdaten auf Unterraum
A_0_ur_dim1  = A_0_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 0
A_1_ur_dim1  = A_1_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 1
A_2_ur_dim1  = A_2_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 2
A_3_ur_dim1  = A_3_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 3
A_4_ur_dim1  = A_4_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 4
A_5_ur_dim1  = A_5_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 5
A_6_ur_dim1  = A_6_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 6
A_7_ur_dim1  = A_7_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 7
A_8_ur_dim1  = A_8_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 8
A_9_ur_dim1  = A_9_nl * pca_ur_1dim; % Datenpunkte fuer Zuglinie 9

% Abbidung der Testdaten auf Unterraum
B_ur_dim1  = B_nl * pca_ur_1dim;

% Erwartungswerte bestimmen
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

% Kovarianzmatrixen bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim1 = [];
for index = 1:size(B_ur_dim1,1)
    
    trainData = B_ur_dim1(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_0_ur_dim1, CVM_A_0_ur_dim1) * A_x_apriori;
    A_1_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_1_ur_dim1, CVM_A_1_ur_dim1) * A_x_apriori;
    A_2_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_2_ur_dim1, CVM_A_2_ur_dim1) * A_x_apriori;
    A_3_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_3_ur_dim1, CVM_A_3_ur_dim1) * A_x_apriori;
    A_4_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_4_ur_dim1, CVM_A_4_ur_dim1) * A_x_apriori;
    A_5_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_5_ur_dim1, CVM_A_5_ur_dim1) * A_x_apriori;
    A_6_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_6_ur_dim1, CVM_A_6_ur_dim1) * A_x_apriori;
    A_7_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_7_ur_dim1, CVM_A_7_ur_dim1) * A_x_apriori;
    A_8_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_8_ur_dim1, CVM_A_8_ur_dim1) * A_x_apriori;
    A_9_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_9_ur_dim1, CVM_A_9_ur_dim1) * A_x_apriori;
    
     % L2 Norm der aposteriori Vorhersage
    A0_l2_dim1 = norm(A_0_aposteriori_predict_dim1);
    A1_l2_dim1 = norm(A_1_aposteriori_predict_dim1);
    A2_l2_dim1 = norm(A_2_aposteriori_predict_dim1);
    A3_l2_dim1 = norm(A_3_aposteriori_predict_dim1);
    A4_l2_dim1 = norm(A_4_aposteriori_predict_dim1);
    A5_l2_dim1 = norm(A_5_aposteriori_predict_dim1);
    A6_l2_dim1 = norm(A_6_aposteriori_predict_dim1);
    A7_l2_dim1 = norm(A_7_aposteriori_predict_dim1);
    A8_l2_dim1 = norm(A_8_aposteriori_predict_dim1);
    A9_l2_dim1 = norm(A_9_aposteriori_predict_dim1);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([A0_l2_dim1, A1_l2_dim1, A2_l2_dim1, A3_l2_dim1, A4_l2_dim1, A5_l2_dim1, A6_l2_dim1, A7_l2_dim1, A8_l2_dim1, A9_l2_dim1]);
        
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
    if (maxValue == norm(A_0_aposteriori_predict_dim1))       % train 0 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),0];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    elseif (maxValue == norm(A_1_aposteriori_predict_dim1))   % train 1 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),1];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    elseif (maxValue == norm(A_2_aposteriori_predict_dim1))   % train 2 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),2];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    elseif (maxValue == norm(A_3_aposteriori_predict_dim1))   % train 3 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),3];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    elseif (maxValue == norm(A_4_aposteriori_predict_dim1))   % train 4 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),4];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    elseif (maxValue == norm(A_5_aposteriori_predict_dim1))   % train 5 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),5];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    elseif (maxValue == norm(A_6_aposteriori_predict_dim1))   % train 6 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),6];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    elseif (maxValue == norm(A_7_aposteriori_predict_dim1))   % train 7 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),7];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    elseif (maxValue == norm(A_8_aposteriori_predict_dim1))   % train 8 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),8];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    else                                                      % train 9 predicted
        tmpVector = [B_ur_dim1(index,:),B(index,B_n),9];
        M_classify_dim1 = vertcat(M_classify_dim1,tmpVector);
    end % end-if
end % end-for_each

M_classify_dim1_n = size(M_classify_dim1,2);
M_classify_dim1_m = size(M_classify_dim1,1);

% Konfusionsmatrix
knownClass_dim1 = M_classify_dim1(:, M_classify_dim1_n -1);
predictedClass_dim1 = M_classify_dim1(:, M_classify_dim1_n);
confusionmatrix_dim1 = confusionmat(knownClass, predictedClass)

%  confusionmatrix_dim1 =

%  341     0     0     0     0     0     0     0    22     0
%    0   350    12     0     1     0     0     0     1     0
%    0     8   355     0     0     0     0     1     0     0
%    0     9     0   320     0     1     0     1     0     5
%    0     0     0     0   362     0     0     0     0     2
%    0     0     0     1     0   323     0     0     2     9
%    0     0     0     0     0     0   325     0    11     0
%    0    28     0     0     0     0     0   314     5    17
%    0     0     0     0     0     0     0     0   336     0
%    0     5     0     0     0     0     0     1     1   329

% Klassifikationsguete
corret_predicted_dim1 = 0;
for index = 1:M_classify_dim1_m
    if M_classify_dim1(index, M_classify_dim1_n -1) == M_classify_dim1(index, M_classify_dim1_n)
        corret_predicted_dim1 = corret_predicted_dim1 + 1;
    end
end
classification_quality_dim1 = corret_predicted_dim1 / M_classify_dim1_m

%  classification_quality_dim1 = 0.4042

%%%%% PCA: 2-dimmensional %%%%%

% Unterraum erzeugen
pca_ur_2dim  = EigVec_CVM_A(:,biggestEVec -1:biggestEVec); % groeßter (= letzter) Eigenvektor

% Abbildung der Trainingsdaten auf Unterraum
A_0_ur_dim2  = A_0_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 0
A_1_ur_dim2  = A_1_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 1
A_2_ur_dim2  = A_2_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 2
A_3_ur_dim2  = A_3_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 3
A_4_ur_dim2  = A_4_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 4
A_5_ur_dim2  = A_5_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 5
A_6_ur_dim2  = A_6_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 6
A_7_ur_dim2  = A_7_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 7
A_8_ur_dim2  = A_8_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 8
A_9_ur_dim2  = A_9_nl * pca_ur_2dim; % Datenpunkte fuer Zuglinie 9

% Abbidung der Testdaten auf Unterraum
B_ur_dim2  = B_nl * pca_ur_2dim;

% Erwartungswerte bestimmen
E_A_0_ur_dim2 = mean(A_0_ur_dim2);
E_A_1_ur_dim2 = mean(A_1_ur_dim2);
E_A_2_ur_dim2 = mean(A_2_ur_dim2);
E_A_3_ur_dim2 = mean(A_3_ur_dim2);
E_A_4_ur_dim2 = mean(A_4_ur_dim2);
E_A_5_ur_dim2 = mean(A_5_ur_dim2);
E_A_6_ur_dim2 = mean(A_6_ur_dim2);
E_A_7_ur_dim2 = mean(A_7_ur_dim2);
E_A_8_ur_dim2 = mean(A_8_ur_dim2);
E_A_9_ur_dim2 = mean(A_9_ur_dim2);

% Kovarianzmatrixen bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim2 = [];
for index = 1:size(B_ur_dim2,1)
    
    trainData = B_ur_dim2(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_0_ur_dim2, CVM_A_0_ur_dim2) * A_x_apriori;
    A_1_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_1_ur_dim2, CVM_A_1_ur_dim2) * A_x_apriori;
    A_2_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_2_ur_dim2, CVM_A_2_ur_dim2) * A_x_apriori;
    A_3_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_3_ur_dim2, CVM_A_3_ur_dim2) * A_x_apriori;
    A_4_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_4_ur_dim2, CVM_A_4_ur_dim2) * A_x_apriori;
    A_5_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_5_ur_dim2, CVM_A_5_ur_dim2) * A_x_apriori;
    A_6_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_6_ur_dim2, CVM_A_6_ur_dim2) * A_x_apriori;
    A_7_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_7_ur_dim2, CVM_A_7_ur_dim2) * A_x_apriori;
    A_8_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_8_ur_dim2, CVM_A_8_ur_dim2) * A_x_apriori;
    A_9_aposteriori_predict_dim2 = mvnpdf(trainData, E_A_9_ur_dim2, CVM_A_9_ur_dim2) * A_x_apriori;
    
    % L2 Norm der aposteriori Vorhersage
    A0_l2_dim2 = norm(A_0_aposteriori_predict_dim2);
    A1_l2_dim2 = norm(A_1_aposteriori_predict_dim2);
    A2_l2_dim2 = norm(A_2_aposteriori_predict_dim2);
    A3_l2_dim2 = norm(A_3_aposteriori_predict_dim2);
    A4_l2_dim2 = norm(A_4_aposteriori_predict_dim2);
    A5_l2_dim2 = norm(A_5_aposteriori_predict_dim2);
    A6_l2_dim2 = norm(A_6_aposteriori_predict_dim2);
    A7_l2_dim2 = norm(A_7_aposteriori_predict_dim2);
    A8_l2_dim2 = norm(A_8_aposteriori_predict_dim2);
    A9_l2_dim2 = norm(A_9_aposteriori_predict_dim2);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([A0_l2_dim2, A1_l2_dim2, A2_l2_dim2, A3_l2_dim2, A4_l2_dim2, A5_l2_dim2, A6_l2_dim2, A7_l2_dim2, A8_l2_dim2, A9_l2_dim2]);
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
    if (maxValue == A0_l2_dim2)       % train 0 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),0];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    elseif (maxValue == A1_l2_dim2)   % train 1 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),1];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    elseif (maxValue == A2_l2_dim2)   % train 2 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),2];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    elseif (maxValue == A3_l2_dim2)   % train 3 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),3];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    elseif (maxValue == A4_l2_dim2)   % train 4 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),4];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    elseif (maxValue == A5_l2_dim2)   % train 5 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),5];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    elseif (maxValue == A6_l2_dim2)   % train 6 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),6];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    elseif (maxValue == A7_l2_dim2)   % train 7 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),7];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    elseif (maxValue == A8_l2_dim2)   % train 8 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),8];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    else                              % train 9 predicted
        tmpVector = [B_ur_dim2(index,:),B(index,B_n),9];
        M_classify_dim2 = vertcat(M_classify_dim2,tmpVector);
    end % end-if
end % end-for_each

M_classify_dim2_n = size(M_classify_dim2,2);
M_classify_dim2_m = size(M_classify_dim2,1);

% Konfusionsmatrix
knownClass_dim2 = M_classify_dim2(:, M_classify_dim2_n -1);
predictedClass_dim2 = M_classify_dim2(:, M_classify_dim2_n);
confusionmatrix_dim2 = confusionmat(knownClass, predictedClass)

%  confusionmatrix_dim2 =
%
%  341     0     0     0     0     0     0     0    22     0
%    0   350    12     0     1     0     0     0     1     0
%    0     8   355     0     0     0     0     1     0     0
%    0     9     0   320     0     1     0     1     0     5
%    0     0     0     0   362     0     0     0     0     2
%    0     0     0     1     0   323     0     0     2     9
%    0     0     0     0     0     0   325     0    11     0
%    0    28     0     0     0     0     0   314     5    17
%    0     0     0     0     0     0     0     0   336     0
%    0     5     0     0     0     0     0     1     1   329

% Klassifikationsguete
corret_predicted_dim2 = 0;
for index = 1:M_classify_dim2_m
    if M_classify_dim2(index, M_classify_dim2_n -1) == M_classify_dim2(index, M_classify_dim2_n)
        corret_predicted_dim2 = corret_predicted_dim2 + 1;
    end
end
classification_quality_dim2 = corret_predicted_dim2 / M_classify_dim2_m

%  classification_quality_dim2 = 0.6515

%%%%% PCA: 3-dimmensional %%%%%

% Unterraum erzeugen
pca_ur_3dim  = EigVec_CVM_A(:,biggestEVec -2:biggestEVec); % groeßter (= letzter) Eigenvektor

% Abbildung der Trainingsdaten auf Unterraum
A_0_ur_dim3  = A_0_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 0
A_1_ur_dim3  = A_1_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 1
A_2_ur_dim3  = A_2_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 2
A_3_ur_dim3  = A_3_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 3
A_4_ur_dim3  = A_4_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 4
A_5_ur_dim3  = A_5_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 5
A_6_ur_dim3  = A_6_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 6
A_7_ur_dim3  = A_7_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 7
A_8_ur_dim3  = A_8_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 8
A_9_ur_dim3  = A_9_nl * pca_ur_3dim; % Datenpunkte fuer Zuglinie 9

% Abbidung der Testdaten auf Unterraum
B_ur_dim3  = B_nl * pca_ur_3dim;

% Erwartungswerte bestimmen
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

% Kovarianzmatrixen bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim3 = [];
for index = 1:size(B_ur_dim3,1)
    
    trainData = B_ur_dim3(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_0_ur_dim3, CVM_A_0_ur_dim3) * A_x_apriori;
    A_1_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_1_ur_dim3, CVM_A_1_ur_dim3) * A_x_apriori;
    A_2_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_2_ur_dim3, CVM_A_2_ur_dim3) * A_x_apriori;
    A_3_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_3_ur_dim3, CVM_A_3_ur_dim3) * A_x_apriori;
    A_4_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_4_ur_dim3, CVM_A_4_ur_dim3) * A_x_apriori;
    A_5_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_5_ur_dim3, CVM_A_5_ur_dim3) * A_x_apriori;
    A_6_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_6_ur_dim3, CVM_A_6_ur_dim3) * A_x_apriori;
    A_7_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_7_ur_dim3, CVM_A_7_ur_dim3) * A_x_apriori;
    A_8_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_8_ur_dim3, CVM_A_8_ur_dim3) * A_x_apriori;
    A_9_aposteriori_predict_dim3 = mvnpdf(trainData, E_A_9_ur_dim3, CVM_A_9_ur_dim3) * A_x_apriori;
    
    % L2 Norm der aposteriori Vorhersage
    A0_l2_dim3 = norm(A_0_aposteriori_predict_dim3);
    A1_l2_dim3 = norm(A_1_aposteriori_predict_dim3);
    A2_l2_dim3 = norm(A_2_aposteriori_predict_dim3);
    A3_l2_dim3 = norm(A_3_aposteriori_predict_dim3);
    A4_l2_dim3 = norm(A_4_aposteriori_predict_dim3);
    A5_l2_dim3 = norm(A_5_aposteriori_predict_dim3);
    A6_l2_dim3 = norm(A_6_aposteriori_predict_dim3);
    A7_l2_dim3 = norm(A_7_aposteriori_predict_dim3);
    A8_l2_dim3 = norm(A_8_aposteriori_predict_dim3);
    A9_l2_dim3 = norm(A_9_aposteriori_predict_dim3);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([A0_l2_dim3, A1_l2_dim3, A2_l2_dim3, A3_l2_dim3, A4_l2_dim3, A5_l2_dim3, A6_l2_dim3, A7_l2_dim3, A8_l2_dim3, A9_l2_dim3]);
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
    if (maxValue == A0_l2_dim3)       % train 0 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),0];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    elseif (maxValue == A1_l2_dim3)   % train 1 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),1];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    elseif (maxValue == A2_l2_dim3)   % train 2 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),2];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    elseif (maxValue == A3_l2_dim3)   % train 3 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),3];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    elseif (maxValue == A4_l2_dim3)   % train 4 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),4];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    elseif (maxValue == A5_l2_dim3)   % train 5 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),5];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    elseif (maxValue == A6_l2_dim3)   % train 6 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),6];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    elseif (maxValue == A7_l2_dim3)   % train 7 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),7];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    elseif (maxValue == A8_l2_dim3)   % train 8 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),8];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    else                              % train 9 predicted
        tmpVector = [B_ur_dim3(index,:),B(index,B_n),9];
        M_classify_dim3 = vertcat(M_classify_dim3,tmpVector);
    end % end-if
end % end-for_each

M_classify_dim3_n = size(M_classify_dim3,2);
M_classify_dim3_m = size(M_classify_dim3,1);

% Konfusionsmatrix
knownClass_dim3 = M_classify_dim3(:, M_classify_dim3_n -1);
predictedClass_dim3 = M_classify_dim3(:, M_classify_dim3_n);
confusionmatrix_dim3 = confusionmat(knownClass, predictedClass)

%  confusionmatrix_dim3 =
%
%  341     0     0     0     0     0     0     0    22     0
%    0   350    12     0     1     0     0     0     1     0
%    0     8   355     0     0     0     0     1     0     0
%    0     9     0   320     0     1     0     1     0     5
%    0     0     0     0   362     0     0     0     0     2
%    0     0     0     1     0   323     0     0     2     9
%    0     0     0     0     0     0   325     0    11     0
%    0    28     0     0     0     0     0   314     5    17
%    0     0     0     0     0     0     0     0   336     0
%    0     5     0     0     0     0     0     1     1   329

% Klassifikationsguete
corret_predicted_dim3 = 0;
for index = 1:M_classify_dim3_m
    if M_classify_dim3(index, M_classify_dim3_n -1) == M_classify_dim3(index, M_classify_dim3_n)
        corret_predicted_dim3 = corret_predicted_dim3 + 1;
    end
end
classification_quality_dim3 = corret_predicted_dim3 / M_classify_dim3_m

%  classification_quality_dim2 = 0.7882

%%%%% PCA: 4-dimmensional %%%%%

% Unterraum erzeugen
pca_ur_4dim  = EigVec_CVM_A(:,biggestEVec -3:biggestEVec);

% Abbildung der Trainingsdaten auf Unterraum
A_0_ur_dim4  = A_0_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 0
A_1_ur_dim4  = A_1_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 1
A_2_ur_dim4  = A_2_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 2
A_3_ur_dim4  = A_3_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 3
A_4_ur_dim4  = A_4_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 4
A_5_ur_dim4  = A_5_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 5
A_6_ur_dim4  = A_6_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 6
A_7_ur_dim4  = A_7_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 7
A_8_ur_dim4  = A_8_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 8
A_9_ur_dim4  = A_9_nl * pca_ur_4dim; % Datenpunkte fuer Zuglinie 9

% Abbidung der Testdaten auf Unterraum
B_ur_dim4  = B_nl * pca_ur_4dim;

% Erwartungswerte bestimmen
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

% Kovarianzmatrixen bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim4 = [];
for index = 1:size(B_ur_dim4,1)
    
    trainData = B_ur_dim4(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_0_ur_dim4, CVM_A_0_ur_dim4) * A_x_apriori;
    A_1_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_1_ur_dim4, CVM_A_1_ur_dim4) * A_x_apriori;
    A_2_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_2_ur_dim4, CVM_A_2_ur_dim4) * A_x_apriori;
    A_3_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_3_ur_dim4, CVM_A_3_ur_dim4) * A_x_apriori;
    A_4_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_4_ur_dim4, CVM_A_4_ur_dim4) * A_x_apriori;
    A_5_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_5_ur_dim4, CVM_A_5_ur_dim4) * A_x_apriori;
    A_6_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_6_ur_dim4, CVM_A_6_ur_dim4) * A_x_apriori;
    A_7_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_7_ur_dim4, CVM_A_7_ur_dim4) * A_x_apriori;
    A_8_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_8_ur_dim4, CVM_A_8_ur_dim4) * A_x_apriori;
    A_9_aposteriori_predict_dim4 = mvnpdf(trainData, E_A_9_ur_dim4, CVM_A_9_ur_dim4) * A_x_apriori;
    
    % L2 Norm der aposteriori Vorhersage
    A0_l2_dim4 = norm(A_0_aposteriori_predict_dim4);
    A1_l2_dim4 = norm(A_1_aposteriori_predict_dim4);
    A2_l2_dim4 = norm(A_2_aposteriori_predict_dim4);
    A3_l2_dim4 = norm(A_3_aposteriori_predict_dim4);
    A4_l2_dim4 = norm(A_4_aposteriori_predict_dim4);
    A5_l2_dim4 = norm(A_5_aposteriori_predict_dim4);
    A6_l2_dim4 = norm(A_6_aposteriori_predict_dim4);
    A7_l2_dim4 = norm(A_7_aposteriori_predict_dim4);
    A8_l2_dim4 = norm(A_8_aposteriori_predict_dim4);
    A9_l2_dim4 = norm(A_9_aposteriori_predict_dim4);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([A0_l2_dim4, A1_l2_dim4, A2_l2_dim4, A3_l2_dim4, A4_l2_dim4, A5_l2_dim4, A6_l2_dim4, A7_l2_dim4, A8_l2_dim4, A9_l2_dim4]);
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
    if (maxValue == A0_l2_dim4)       % train 0 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),0];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    elseif (maxValue == A1_l2_dim4)   % train 1 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),1];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    elseif (maxValue == A2_l2_dim4)   % train 2 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),2];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    elseif (maxValue == A3_l2_dim4)   % train 3 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),3];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    elseif (maxValue == A4_l2_dim4)   % train 4 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),4];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    elseif (maxValue == A5_l2_dim4)   % train 5 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),5];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    elseif (maxValue == A6_l2_dim4)   % train 6 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),6];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    elseif (maxValue == A7_l2_dim4)   % train 7 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),7];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    elseif (maxValue == A8_l2_dim4)   % train 8 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),8];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    else                              % train 9 predicted
        tmpVector = [B_ur_dim4(index,:),B(index,B_n),9];
        M_classify_dim4 = vertcat(M_classify_dim4,tmpVector);
    end % end-if
end % end-for_each

M_classify_dim4_n = size(M_classify_dim4,2);
M_classify_dim4_m = size(M_classify_dim4,1);

% Konfusionsmatrix
knownClass_dim4 = M_classify_dim4(:, M_classify_dim4_n -1);
predictedClass_dim4 = M_classify_dim4(:, M_classify_dim4_n);
confusionmatrix_dim4 = confusionmat(knownClass, predictedClass)

%  confusionmatrix_dim4 =
%
%  341     0     0     0     0     0     0     0    22     0
%    0   350    12     0     1     0     0     0     1     0
%    0     8   355     0     0     0     0     1     0     0
%    0     9     0   320     0     1     0     1     0     5
%    0     0     0     0   362     0     0     0     0     2
%    0     0     0     1     0   323     0     0     2     9
%    0     0     0     0     0     0   325     0    11     0
%    0    28     0     0     0     0     0   314     5    17
%    0     0     0     0     0     0     0     0   336     0
%    0     5     0     0     0     0     0     1     1   329
     
% Klassifikationsguete
corret_predicted_dim4 = 0;
for index = 1:M_classify_dim4_m
    if M_classify_dim4(index, M_classify_dim4_n -1) == M_classify_dim4(index, M_classify_dim4_n)
        corret_predicted_dim4 = corret_predicted_dim4 + 1;
    end
end
classification_quality_dim4 = corret_predicted_dim4 / M_classify_dim4_m

%  classification_quality_dim4 = 0.8382

%%%%% PCA: 5-dimmensional %%%%%

% Unterraum erzeugen
pca_ur_5dim  = EigVec_CVM_A(:,biggestEVec -4:biggestEVec);

% Abbildung der Trainingsdaten auf Unterraum
A_0_ur_dim5  = A_0_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 0
A_1_ur_dim5  = A_1_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 1
A_2_ur_dim5  = A_2_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 2
A_3_ur_dim5  = A_3_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 3
A_4_ur_dim5  = A_4_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 4
A_5_ur_dim5  = A_5_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 5
A_6_ur_dim5  = A_6_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 6
A_7_ur_dim5  = A_7_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 7
A_8_ur_dim5  = A_8_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 8
A_9_ur_dim5  = A_9_nl * pca_ur_5dim; % Datenpunkte fuer Zuglinie 9

% Abbidung der Testdaten auf Unterraum
B_ur_dim5  = B_nl * pca_ur_5dim;

% Erwartungswerte bestimmen
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

% Kovarianzmatrixen bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim5 = [];
for index = 1:size(B_ur_dim5,1)
    
    trainData = B_ur_dim5(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_0_ur_dim5, CVM_A_0_ur_dim5) * A_x_apriori;
    A_1_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_1_ur_dim5, CVM_A_1_ur_dim5) * A_x_apriori;
    A_2_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_2_ur_dim5, CVM_A_2_ur_dim5) * A_x_apriori;
    A_3_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_3_ur_dim5, CVM_A_3_ur_dim5) * A_x_apriori;
    A_4_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_4_ur_dim5, CVM_A_4_ur_dim5) * A_x_apriori;
    A_5_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_5_ur_dim5, CVM_A_5_ur_dim5) * A_x_apriori;
    A_6_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_6_ur_dim5, CVM_A_6_ur_dim5) * A_x_apriori;
    A_7_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_7_ur_dim5, CVM_A_7_ur_dim5) * A_x_apriori;
    A_8_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_8_ur_dim5, CVM_A_8_ur_dim5) * A_x_apriori;
    A_9_aposteriori_predict_dim5 = mvnpdf(trainData, E_A_9_ur_dim5, CVM_A_9_ur_dim5) * A_x_apriori;
    
    % L2 Norm der aposteriori Vorhersage
    A0_l2_dim5 = norm(A_0_aposteriori_predict_dim5);
    A1_l2_dim5 = norm(A_1_aposteriori_predict_dim5);
    A2_l2_dim5 = norm(A_2_aposteriori_predict_dim5);
    A3_l2_dim5 = norm(A_3_aposteriori_predict_dim5);
    A4_l2_dim5 = norm(A_4_aposteriori_predict_dim5);
    A5_l2_dim5 = norm(A_5_aposteriori_predict_dim5);
    A6_l2_dim5 = norm(A_6_aposteriori_predict_dim5);
    A7_l2_dim5 = norm(A_7_aposteriori_predict_dim5);
    A8_l2_dim5 = norm(A_8_aposteriori_predict_dim5);
    A9_l2_dim5 = norm(A_9_aposteriori_predict_dim5);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([A0_l2_dim5, A1_l2_dim5, A2_l2_dim5, A3_l2_dim5, A4_l2_dim5, A5_l2_dim5, A6_l2_dim5, A7_l2_dim5, A8_l2_dim5, A9_l2_dim5]);
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
    if (maxValue == A0_l2_dim5)       % train 0 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),0];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    elseif (maxValue == A1_l2_dim5)   % train 1 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),1];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    elseif (maxValue == A2_l2_dim5)   % train 2 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),2];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    elseif (maxValue == A3_l2_dim5)   % train 3 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),3];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    elseif (maxValue == A4_l2_dim5)   % train 4 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),4];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    elseif (maxValue == A5_l2_dim5)   % train 5 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),5];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    elseif (maxValue == A6_l2_dim5)   % train 6 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),6];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    elseif (maxValue == A7_l2_dim5)   % train 7 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),7];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    elseif (maxValue == A8_l2_dim5)   % train 8 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),8];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    else                              % train 9 predicted
        tmpVector = [B_ur_dim5(index,:),B(index,B_n),9];
        M_classify_dim5 = vertcat(M_classify_dim5,tmpVector);
    end % end-if
end % end-for_each

M_classify_dim5_n = size(M_classify_dim5,2);
M_classify_dim5_m = size(M_classify_dim5,1);

% Konfusionsmatrix
knownClass_dim5 = M_classify_dim5(:, M_classify_dim5_n -1);
predictedClass_dim5 = M_classify_dim5(:, M_classify_dim5_n);
confusionmatrix_dim5 = confusionmat(knownClass, predictedClass)

%  confusionmatrix_dim4 =
%
%  341     0     0     0     0     0     0     0    22     0
%    0   350    12     0     1     0     0     0     1     0
%    0     8   355     0     0     0     0     1     0     0
%    0     9     0   320     0     1     0     1     0     5
%    0     0     0     0   362     0     0     0     0     2
%    0     0     0     1     0   323     0     0     2     9
%    0     0     0     0     0     0   325     0    11     0
%    0    28     0     0     0     0     0   314     5    17
%    0     0     0     0     0     0     0     0   336     0
%    0     5     0     0     0     0     0     1     1   329

% Klassifikationsguete
corret_predicted_dim5 = 0;
for index = 1:M_classify_dim5_m
    if M_classify_dim5(index, M_classify_dim5_n -1) == M_classify_dim5(index, M_classify_dim5_n)
        corret_predicted_dim5 = corret_predicted_dim5 + 1;
    end
end
classification_quality_dim5 = corret_predicted_dim5 / M_classify_dim5_m

%  classification_quality_dim5 = 0.8708

%%%%% PCA: 6-dimmensional %%%%%

% Unterraum erzeugen
pca_ur_6dim  = EigVec_CVM_A(:,biggestEVec -5:biggestEVec);

% Abbildung der Trainingsdaten auf Unterraum
A_0_ur_dim6  = A_0_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 0
A_1_ur_dim6  = A_1_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 1
A_2_ur_dim6  = A_2_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 2
A_3_ur_dim6  = A_3_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 3
A_4_ur_dim6  = A_4_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 4
A_5_ur_dim6  = A_5_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 5
A_6_ur_dim6  = A_6_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 6
A_7_ur_dim6  = A_7_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 7
A_8_ur_dim6  = A_8_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 8
A_9_ur_dim6  = A_9_nl * pca_ur_6dim; % Datenpunkte fuer Zuglinie 9

% Abbidung der Testdaten auf Unterraum
B_ur_dim6  = B_nl * pca_ur_6dim;

% Erwartungswerte bestimmen
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

% Kovarianzmatrixen bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim6 = [];
for index = 1:size(B_ur_dim6,1)
    
    trainData = B_ur_dim6(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_0_ur_dim6, CVM_A_0_ur_dim6) * A_x_apriori;
    A_1_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_1_ur_dim6, CVM_A_1_ur_dim6) * A_x_apriori;
    A_2_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_2_ur_dim6, CVM_A_2_ur_dim6) * A_x_apriori;
    A_3_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_3_ur_dim6, CVM_A_3_ur_dim6) * A_x_apriori;
    A_4_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_4_ur_dim6, CVM_A_4_ur_dim6) * A_x_apriori;
    A_5_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_5_ur_dim6, CVM_A_5_ur_dim6) * A_x_apriori;
    A_6_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_6_ur_dim6, CVM_A_6_ur_dim6) * A_x_apriori;
    A_7_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_7_ur_dim6, CVM_A_7_ur_dim6) * A_x_apriori;
    A_8_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_8_ur_dim6, CVM_A_8_ur_dim6) * A_x_apriori;
    A_9_aposteriori_predict_dim6 = mvnpdf(trainData, E_A_9_ur_dim6, CVM_A_9_ur_dim6) * A_x_apriori;
    
    % L2 Norm der aposteriori Vorhersage
    A0_l2_dim6 = norm(A_0_aposteriori_predict_dim6);
    A1_l2_dim6 = norm(A_1_aposteriori_predict_dim6);
    A2_l2_dim6 = norm(A_2_aposteriori_predict_dim6);
    A3_l2_dim6 = norm(A_3_aposteriori_predict_dim6);
    A4_l2_dim6 = norm(A_4_aposteriori_predict_dim6);
    A5_l2_dim6 = norm(A_5_aposteriori_predict_dim6);
    A6_l2_dim6 = norm(A_6_aposteriori_predict_dim6);
    A7_l2_dim6 = norm(A_7_aposteriori_predict_dim6);
    A8_l2_dim6 = norm(A_8_aposteriori_predict_dim6);
    A9_l2_dim6 = norm(A_9_aposteriori_predict_dim6);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([A0_l2_dim6, A1_l2_dim6, A2_l2_dim6, A3_l2_dim6, A4_l2_dim6, A5_l2_dim6, A6_l2_dim6, A7_l2_dim6, A8_l2_dim6, A9_l2_dim6]);
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
    if (maxValue == A0_l2_dim6)       % train 0 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),0];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    elseif (maxValue == A1_l2_dim6)   % train 1 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),1];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    elseif (maxValue == A2_l2_dim6)   % train 2 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),2];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    elseif (maxValue == A3_l2_dim6)   % train 3 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),3];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    elseif (maxValue == A4_l2_dim6)   % train 4 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),4];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    elseif (maxValue == A5_l2_dim6)   % train 5 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),5];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    elseif (maxValue == A6_l2_dim6)   % train 6 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),6];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    elseif (maxValue == A7_l2_dim6)   % train 7 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),7];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    elseif (maxValue == A8_l2_dim6)   % train 8 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),8];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    else                              % train 9 predicted
        tmpVector = [B_ur_dim6(index,:),B(index,B_n),9];
        M_classify_dim6 = vertcat(M_classify_dim6,tmpVector);
    end % end-if
end % end-for_each

M_classify_dim6_n = size(M_classify_dim6,2);
M_classify_dim6_m = size(M_classify_dim6,1);

% Konfusionsmatrix
knownClass_dim6 = M_classify_dim6(:, M_classify_dim6_n -1);
predictedClass_dim6 = M_classify_dim6(:, M_classify_dim6_n);
confusionmatrix_dim6 = confusionmat(knownClass, predictedClass)

%  confusionmatrix_dim6 =
%
%  341     0     0     0     0     0     0     0    22     0
%    0   350    12     0     1     0     0     0     1     0
%    0     8   355     0     0     0     0     1     0     0
%    0     9     0   320     0     1     0     1     0     5
%    0     0     0     0   362     0     0     0     0     2
%    0     0     0     1     0   323     0     0     2     9
%    0     0     0     0     0     0   325     0    11     0
%    0    28     0     0     0     0     0   314     5    17
%    0     0     0     0     0     0     0     0   336     0
%    0     5     0     0     0     0     0     1     1   329

% Klassifikationsguete
corret_predicted_dim6 = 0;
for index = 1:M_classify_dim6_m
    if M_classify_dim6(index, M_classify_dim6_n -1) == M_classify_dim6(index, M_classify_dim6_n)
        corret_predicted_dim6 = corret_predicted_dim6 + 1;
    end
end
classification_quality_dim6 = corret_predicted_dim6 / M_classify_dim6_m

%  classification_quality_dim6 = 0.8957

%%%%% PCA: 7-dimmensional %%%%%

% Unterraum erzeugen
pca_ur_7dim  = EigVec_CVM_A(:,biggestEVec -6:biggestEVec);

% Abbildung der Trainingsdaten auf Unterraum
A_0_ur_dim7  = A_0_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 0
A_1_ur_dim7  = A_1_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 1
A_2_ur_dim7  = A_2_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 2
A_3_ur_dim7  = A_3_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 3
A_4_ur_dim7  = A_4_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 4
A_5_ur_dim7  = A_5_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 5
A_6_ur_dim7  = A_6_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 6
A_7_ur_dim7  = A_7_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 7
A_8_ur_dim7  = A_8_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 8
A_9_ur_dim7  = A_9_nl * pca_ur_7dim; % Datenpunkte fuer Zuglinie 9

% Abbidung der Testdaten auf Unterraum
B_ur_dim7  = B_nl * pca_ur_7dim;

% Erwartungswerte bestimmen
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

% Kovarianzmatrixen bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim7 = [];
for index = 1:size(B_ur_dim7,1)
    
    trainData = B_ur_dim7(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_0_ur_dim7, CVM_A_0_ur_dim7) * A_x_apriori;
    A_1_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_1_ur_dim7, CVM_A_1_ur_dim7) * A_x_apriori;
    A_2_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_2_ur_dim7, CVM_A_2_ur_dim7) * A_x_apriori;
    A_3_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_3_ur_dim7, CVM_A_3_ur_dim7) * A_x_apriori;
    A_4_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_4_ur_dim7, CVM_A_4_ur_dim7) * A_x_apriori;
    A_5_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_5_ur_dim7, CVM_A_5_ur_dim7) * A_x_apriori;
    A_6_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_6_ur_dim7, CVM_A_6_ur_dim7) * A_x_apriori;
    A_7_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_7_ur_dim7, CVM_A_7_ur_dim7) * A_x_apriori;
    A_8_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_8_ur_dim7, CVM_A_8_ur_dim7) * A_x_apriori;
    A_9_aposteriori_predict_dim7 = mvnpdf(trainData, E_A_9_ur_dim7, CVM_A_9_ur_dim7) * A_x_apriori;
    
    % L2 Norm der aposteriori Vorhersage
    A0_l2_dim7 = norm(A_0_aposteriori_predict_dim7);
    A1_l2_dim7 = norm(A_1_aposteriori_predict_dim7);
    A2_l2_dim7 = norm(A_2_aposteriori_predict_dim7);
    A3_l2_dim7 = norm(A_3_aposteriori_predict_dim7);
    A4_l2_dim7 = norm(A_4_aposteriori_predict_dim7);
    A5_l2_dim7 = norm(A_5_aposteriori_predict_dim7);
    A6_l2_dim7 = norm(A_6_aposteriori_predict_dim7);
    A7_l2_dim7 = norm(A_7_aposteriori_predict_dim7);
    A8_l2_dim7 = norm(A_8_aposteriori_predict_dim7);
    A9_l2_dim7 = norm(A_9_aposteriori_predict_dim7);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([A0_l2_dim7, A1_l2_dim7, A2_l2_dim7, A3_l2_dim7, A4_l2_dim7, A5_l2_dim7, A6_l2_dim7, A7_l2_dim7, A8_l2_dim7, A9_l2_dim7]);
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
    if (maxValue == A0_l2_dim7)       % train 0 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),0];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    elseif (maxValue == A1_l2_dim7)   % train 1 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),1];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    elseif (maxValue == A2_l2_dim7)   % train 2 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),2];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    elseif (maxValue == A3_l2_dim7)   % train 3 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),3];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    elseif (maxValue == A4_l2_dim7)   % train 4 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),4];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    elseif (maxValue == A5_l2_dim7)   % train 5 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),5];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    elseif (maxValue == A6_l2_dim7)   % train 6 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),6];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    elseif (maxValue == A7_l2_dim7)   % train 7 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),7];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    elseif (maxValue == A8_l2_dim7)   % train 8 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),8];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    else                              % train 9 predicted
        tmpVector = [B_ur_dim7(index,:),B(index,B_n),9];
        M_classify_dim7 = vertcat(M_classify_dim7,tmpVector);
    end % end-if
end % end-for_each

M_classify_dim7_n = size(M_classify_dim7,2);
M_classify_dim7_m = size(M_classify_dim7,1);

% Konfusionsmatrix
knownClass_dim7 = M_classify_dim7(:, M_classify_dim7_n -1);
predictedClass_dim7 = M_classify_dim7(:, M_classify_dim7_n);
confusionmatrix_dim7 = confusionmat(knownClass, predictedClass)

%  confusionmatrix_dim7 =
%
%  341     0     0     0     0     0     0     0    22     0
%    0   350    12     0     1     0     0     0     1     0
%    0     8   355     0     0     0     0     1     0     0
%    0     9     0   320     0     1     0     1     0     5
%    0     0     0     0   362     0     0     0     0     2
%    0     0     0     1     0   323     0     0     2     9
%    0     0     0     0     0     0   325     0    11     0
%    0    28     0     0     0     0     0   314     5    17
%    0     0     0     0     0     0     0     0   336     0
%    0     5     0     0     0     0     0     1     1   329

% Klassifikationsguete
corret_predicted_dim7 = 0;
for index = 1:M_classify_dim7_m
    if M_classify_dim7(index, M_classify_dim7_n -1) == M_classify_dim7(index, M_classify_dim7_n)
        corret_predicted_dim7 = corret_predicted_dim7 + 1;
    end
end
classification_quality_dim7 = corret_predicted_dim7 / M_classify_dim7_m

%  classification_quality_dim7 = 0.9062

%%%%% PCA: 8-dimmensional %%%%%

% Unterraum erzeugen
pca_ur_8dim  = EigVec_CVM_A(:,biggestEVec -7:biggestEVec);

% Abbildung der Trainingsdaten auf Unterraum
A_0_ur_dim8  = A_0_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 0
A_1_ur_dim8  = A_1_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 1
A_2_ur_dim8  = A_2_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 2
A_3_ur_dim8  = A_3_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 3
A_4_ur_dim8  = A_4_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 4
A_5_ur_dim8  = A_5_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 5
A_6_ur_dim8  = A_6_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 6
A_7_ur_dim8  = A_7_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 7
A_8_ur_dim8  = A_8_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 8
A_9_ur_dim8  = A_9_nl * pca_ur_8dim; % Datenpunkte fuer Zuglinie 9

% Abbidung der Testdaten auf Unterraum
B_ur_dim8  = B_nl * pca_ur_8dim;

% Erwartungswerte bestimmen
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

% Kovarianzmatrixen bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim8 = [];
for index = 1:size(B_ur_dim8,1)
    
    trainData = B_ur_dim8(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_0_ur_dim8, CVM_A_0_ur_dim8) * A_x_apriori;
    A_1_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_1_ur_dim8, CVM_A_1_ur_dim8) * A_x_apriori;
    A_2_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_2_ur_dim8, CVM_A_2_ur_dim8) * A_x_apriori;
    A_3_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_3_ur_dim8, CVM_A_3_ur_dim8) * A_x_apriori;
    A_4_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_4_ur_dim8, CVM_A_4_ur_dim8) * A_x_apriori;
    A_5_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_5_ur_dim8, CVM_A_5_ur_dim8) * A_x_apriori;
    A_6_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_6_ur_dim8, CVM_A_6_ur_dim8) * A_x_apriori;
    A_7_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_7_ur_dim8, CVM_A_7_ur_dim8) * A_x_apriori;
    A_8_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_8_ur_dim8, CVM_A_8_ur_dim8) * A_x_apriori;
    A_9_aposteriori_predict_dim8 = mvnpdf(trainData, E_A_9_ur_dim8, CVM_A_9_ur_dim8) * A_x_apriori;
    
    % L2 Norm der aposteriori Vorhersage
    A0_l2_dim8 = norm(A_0_aposteriori_predict_dim8);
    A1_l2_dim8 = norm(A_1_aposteriori_predict_dim8);
    A2_l2_dim8 = norm(A_2_aposteriori_predict_dim8);
    A3_l2_dim8 = norm(A_3_aposteriori_predict_dim8);
    A4_l2_dim8 = norm(A_4_aposteriori_predict_dim8);
    A5_l2_dim8 = norm(A_5_aposteriori_predict_dim8);
    A6_l2_dim8 = norm(A_6_aposteriori_predict_dim8);
    A7_l2_dim8 = norm(A_7_aposteriori_predict_dim8);
    A8_l2_dim8 = norm(A_8_aposteriori_predict_dim8);
    A9_l2_dim8 = norm(A_9_aposteriori_predict_dim8);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([A0_l2_dim8, A1_l2_dim8, A2_l2_dim8, A3_l2_dim8, A4_l2_dim8, A5_l2_dim8, A6_l2_dim8, A7_l2_dim8, A8_l2_dim8, A9_l2_dim8]);
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
    if (maxValue == A0_l2_dim8)       % train 0 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),0];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    elseif (maxValue == A1_l2_dim8)   % train 1 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),1];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    elseif (maxValue == A2_l2_dim8)   % train 2 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),2];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    elseif (maxValue == A3_l2_dim8)   % train 3 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),3];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    elseif (maxValue == A4_l2_dim8)   % train 4 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),4];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    elseif (maxValue == A5_l2_dim8)   % train 5 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),5];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    elseif (maxValue == A6_l2_dim8)   % train 6 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),6];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    elseif (maxValue == A7_l2_dim8)   % train 7 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),7];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    elseif (maxValue == A8_l2_dim8)   % train 8 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),8];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    else                              % train 9 predicted
        tmpVector = [B_ur_dim8(index,:),B(index,B_n),9];
        M_classify_dim8 = vertcat(M_classify_dim8,tmpVector);
    end % end-if
end % end-for_each

M_classify_dim8_n = size(M_classify_dim8,2);
M_classify_dim8_m = size(M_classify_dim8,1);

% Konfusionsmatrix
knownClass_dim8 = M_classify_dim8(:, M_classify_dim8_n -1);
predictedClass_dim8 = M_classify_dim8(:, M_classify_dim8_n);
confusionmatrix_dim8 = confusionmat(knownClass, predictedClass)

%  confusionmatrix_dim8 =
%
%  341     0     0     0     0     0     0     0    22     0
%    0   350    12     0     1     0     0     0     1     0
%    0     8   355     0     0     0     0     1     0     0
%    0     9     0   320     0     1     0     1     0     5
%    0     0     0     0   362     0     0     0     0     2
%    0     0     0     1     0   323     0     0     2     9
%    0     0     0     0     0     0   325     0    11     0
%    0    28     0     0     0     0     0   314     5    17
%    0     0     0     0     0     0     0     0   336     0
%    0     5     0     0     0     0     0     1     1   329

% Klassifikationsguete
corret_predicted_dim8 = 0;
for index = 1:M_classify_dim8_m
    if M_classify_dim8(index, M_classify_dim8_n -1) == M_classify_dim8(index, M_classify_dim8_n)
        corret_predicted_dim8 = corret_predicted_dim8 + 1;
    end
end
classification_quality_dim8 = corret_predicted_dim8 / M_classify_dim8_m

%  classification_quality_dim8 = 0.9260

%%%%% PCA: 9-dimmensional %%%%%

% Unterraum erzeugen
pca_ur_9dim  = EigVec_CVM_A(:,biggestEVec -8:biggestEVec);

% Abbildung der Trainingsdaten auf Unterraum
A_0_ur_dim9  = A_0_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 0
A_1_ur_dim9  = A_1_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 1
A_2_ur_dim9  = A_2_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 2
A_3_ur_dim9  = A_3_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 3
A_4_ur_dim9  = A_4_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 4
A_5_ur_dim9  = A_5_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 5
A_6_ur_dim9  = A_6_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 6
A_7_ur_dim9  = A_7_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 7
A_8_ur_dim9  = A_8_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 8
A_9_ur_dim9  = A_9_nl * pca_ur_9dim; % Datenpunkte fuer Zuglinie 9

% Abbidung der Testdaten auf Unterraum
B_ur_dim9  = B_nl * pca_ur_9dim;

% Erwartungswerte bestimmen
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

% Kovarianzmatrixen bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim9 = [];
for index = 1:size(B_ur_dim9,1)
    
    trainData = B_ur_dim9(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_0_ur_dim9, CVM_A_0_ur_dim9) * A_x_apriori;
    A_1_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_1_ur_dim9, CVM_A_1_ur_dim9) * A_x_apriori;
    A_2_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_2_ur_dim9, CVM_A_2_ur_dim9) * A_x_apriori;
    A_3_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_3_ur_dim9, CVM_A_3_ur_dim9) * A_x_apriori;
    A_4_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_4_ur_dim9, CVM_A_4_ur_dim9) * A_x_apriori;
    A_5_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_5_ur_dim9, CVM_A_5_ur_dim9) * A_x_apriori;
    A_6_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_6_ur_dim9, CVM_A_6_ur_dim9) * A_x_apriori;
    A_7_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_7_ur_dim9, CVM_A_7_ur_dim9) * A_x_apriori;
    A_8_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_8_ur_dim9, CVM_A_8_ur_dim9) * A_x_apriori;
    A_9_aposteriori_predict_dim9 = mvnpdf(trainData, E_A_9_ur_dim9, CVM_A_9_ur_dim9) * A_x_apriori;
    
    % L2 Norm der aposteriori Vorhersage
    A0_l2_dim9 = norm(A_0_aposteriori_predict_dim9);
    A1_l2_dim9 = norm(A_1_aposteriori_predict_dim9);
    A2_l2_dim9 = norm(A_2_aposteriori_predict_dim9);
    A3_l2_dim9 = norm(A_3_aposteriori_predict_dim9);
    A4_l2_dim9 = norm(A_4_aposteriori_predict_dim9);
    A5_l2_dim9 = norm(A_5_aposteriori_predict_dim9);
    A6_l2_dim9 = norm(A_6_aposteriori_predict_dim9);
    A7_l2_dim9 = norm(A_7_aposteriori_predict_dim9);
    A8_l2_dim9 = norm(A_8_aposteriori_predict_dim9);
    A9_l2_dim9 = norm(A_9_aposteriori_predict_dim9);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([A0_l2_dim9, A1_l2_dim9, A2_l2_dim9, A3_l2_dim9, A4_l2_dim9, A5_l2_dim9, A6_l2_dim9, A7_l2_dim9, A8_l2_dim9, A9_l2_dim9]);
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
    if (maxValue == A0_l2_dim9)       % train 0 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),0];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    elseif (maxValue == A1_l2_dim9)   % train 1 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),1];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    elseif (maxValue == A2_l2_dim9)   % train 2 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),2];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    elseif (maxValue == A3_l2_dim9)   % train 3 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),3];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    elseif (maxValue == A4_l2_dim9)   % train 4 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),4];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    elseif (maxValue == A5_l2_dim9)   % train 5 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),5];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    elseif (maxValue == A6_l2_dim9)   % train 6 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),6];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    elseif (maxValue == A7_l2_dim9)   % train 7 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),7];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    elseif (maxValue == A8_l2_dim9)   % train 8 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),8];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    else                              % train 9 predicted
        tmpVector = [B_ur_dim9(index,:),B(index,B_n),9];
        M_classify_dim9 = vertcat(M_classify_dim9,tmpVector);
    end % end-if
end % end-for_each

M_classify_dim9_n = size(M_classify_dim9,2);
M_classify_dim9_m = size(M_classify_dim9,1);

% Konfusionsmatrix
knownClass_dim9 = M_classify_dim9(:, M_classify_dim9_n -1);
predictedClass_dim9 = M_classify_dim9(:, M_classify_dim9_n);
confusionmatrix_dim9 = confusionmat(knownClass, predictedClass)

%  confusionmatrix_dim9 =
%
%  341     0     0     0     0     0     0     0    22     0
%    0   350    12     0     1     0     0     0     1     0
%    0     8   355     0     0     0     0     1     0     0
%    0     9     0   320     0     1     0     1     0     5
%    0     0     0     0   362     0     0     0     0     2
%    0     0     0     1     0   323     0     0     2     9
%    0     0     0     0     0     0   325     0    11     0
%    0    28     0     0     0     0     0   314     5    17
%    0     0     0     0     0     0     0     0   336     0
%    0     5     0     0     0     0     0     1     1   329

% Klassifikationsguete
corret_predicted_dim9 = 0;
for index = 1:M_classify_dim9_m
    if M_classify_dim9(index, M_classify_dim9_n -1) == M_classify_dim9(index, M_classify_dim9_n)
        corret_predicted_dim9 = corret_predicted_dim9 + 1;
    end
end
classification_quality_dim9 = corret_predicted_dim9 / M_classify_dim9_m

%  classification_quality_dim9 = 0.9491