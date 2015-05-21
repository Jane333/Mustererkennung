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
    
    % multivariate PDF für Testdatensatz (für jede Zuglinie)
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
    
    % Bayes Klassifikation (Welche aposteriori Vorhersage war die Größte?)
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

% PCA: 1-dimmensional
biggestEVec = size(EigVec_CVM_A,2);

% Unterraum erzeugen
pca_ur_1dim  = EigVec_CVM_A(:,biggestEVec); % groeßter (= letzter) Eigenvektor

% Abbildung der gesamten Trainingsdaten auf Unterraum
%A_ur_dim1  = A_nl * pca_ur_1dim;

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

% multivariate PDF bestimmen
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

% Aposteriori - Wahrscheinlichkeit bestimmen
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

% Klassifizierung der Testdaten (Metrik: L2-Norm)
M_classify_dim1 = [];
for index = 1:size(B_ur_dim1,1)
    
    trainData = B_ur_dim1(index,:);
    
    % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
    A_0_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_0_ur_dim1, CVM_A_0_ur_dim1);
    A_1_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_1_ur_dim1, CVM_A_1_ur_dim1);
    A_2_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_2_ur_dim1, CVM_A_2_ur_dim1);
    A_3_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_3_ur_dim1, CVM_A_3_ur_dim1);
    A_4_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_4_ur_dim1, CVM_A_4_ur_dim1);
    A_5_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_5_ur_dim1, CVM_A_5_ur_dim1);
    A_6_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_6_ur_dim1, CVM_A_6_ur_dim1);
    A_7_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_7_ur_dim1, CVM_A_7_ur_dim1);
    A_8_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_8_ur_dim1, CVM_A_8_ur_dim1);
    A_9_aposteriori_predict_dim1 = mvnpdf(trainData, E_A_9_ur_dim1, CVM_A_9_ur_dim1);
    
    % Bestimmung des Maximums (aposteriori Vorhersage)
    [maxValue, indexAtMaxValue] = max([norm(A_0_aposteriori_predict_dim1),norm(A_1_aposteriori_predict_dim1),norm(A_2_aposteriori_predict_dim1),norm(A_3_aposteriori_predict_dim1),norm(A_4_aposteriori_predict_dim1),norm(A_5_aposteriori_predict_dim1),norm(A_6_aposteriori_predict_dim1),norm(A_7_aposteriori_predict_dim1),norm(A_8_aposteriori_predict_dim1),norm(A_9_aposteriori_predict_dim1)]);
    
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

% Klassifikationsguete
corret_predicted_dim1 = 0;
for index = 1:M_classify_dim1_m
    if M_classify_dim1(index, M_classify_dim1_n -1) == M_classify_dim1(index, M_classify_dim1_n)
        corret_predicted_dim1 = corret_predicted_dim1 + 1;
    end
end
classification_quality_dim1 = corret_predicted_dim1 / M_classify_dim1_m