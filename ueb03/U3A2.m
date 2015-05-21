%%% 3. Uebung - Aufgabe 2

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

biggestEVec = size(EigVec_CVM_A,2);
X = EigVal_CVM_A(:,[16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1])
A_x_apriori = 1/10;

for dim = [1:16]

    % Unterraum erzeugen
    pca_ur  = X(:,1:dim);
    
    % Abbildung der Trainingsdaten auf Unterraum
    A_0_ur = A_0_nl * pca_ur; % Datenpunkte fuer Zuglinie 0
    A_1_ur = A_1_nl * pca_ur; % Datenpunkte fuer Zuglinie 1
    A_2_ur = A_2_nl * pca_ur; % Datenpunkte fuer Zuglinie 2
    A_3_ur = A_3_nl * pca_ur; % Datenpunkte fuer Zuglinie 3
    A_4_ur = A_4_nl * pca_ur; % Datenpunkte fuer Zuglinie 4
    A_5_ur = A_5_nl * pca_ur; % Datenpunkte fuer Zuglinie 5
    A_6_ur = A_6_nl * pca_ur; % Datenpunkte fuer Zuglinie 6
    A_7_ur = A_7_nl * pca_ur; % Datenpunkte fuer Zuglinie 7
    A_8_ur = A_8_nl * pca_ur; % Datenpunkte fuer Zuglinie 8
    A_9_ur = A_9_nl * pca_ur; % Datenpunkte fuer Zuglinie 9
    
    % Abbidung der Testdaten auf Unterraum
    B_ur  = B_nl * pca_ur;
    
    % Erwartungswerte bestimmen
    E_A_0_ur = mean(A_0_ur);
    E_A_1_ur = mean(A_1_ur);
    E_A_2_ur = mean(A_2_ur);
    E_A_3_ur = mean(A_3_ur);
    E_A_4_ur = mean(A_4_ur);
    E_A_5_ur = mean(A_5_ur);
    E_A_6_ur = mean(A_6_ur);
    E_A_7_ur = mean(A_7_ur);
    E_A_8_ur = mean(A_8_ur);
    E_A_9_ur = mean(A_9_ur);
    
    % Kovarianzmatrixen bestimmen
    CVM_A_0_ur = cov(A_0_ur);
    CVM_A_1_ur = cov(A_1_ur);
    CVM_A_2_ur = cov(A_2_ur);
    CVM_A_3_ur = cov(A_3_ur);
    CVM_A_4_ur = cov(A_4_ur);
    CVM_A_5_ur = cov(A_5_ur);
    CVM_A_6_ur = cov(A_6_ur);
    CVM_A_7_ur = cov(A_7_ur);
    CVM_A_8_ur = cov(A_8_ur);
    CVM_A_9_ur = cov(A_9_ur);
    
    % Klassifizierung der Testdaten (Metrik: L2-Norm)
    M_classify = [];
    for index = 1:size(B_ur,1)
        trainData = B_ur(index,:);
    
        % multivariate PDF fuer Testdatensatz (für jede Zuglinie)
        A_0_aposteriori_predict = mvnpdf(trainData, E_A_0_ur, CVM_A_0_ur) * A_x_apriori;
        A_1_aposteriori_predict = mvnpdf(trainData, E_A_1_ur, CVM_A_1_ur) * A_x_apriori;
        A_2_aposteriori_predict = mvnpdf(trainData, E_A_2_ur, CVM_A_2_ur) * A_x_apriori;
        A_3_aposteriori_predict = mvnpdf(trainData, E_A_3_ur, CVM_A_3_ur) * A_x_apriori;
        A_4_aposteriori_predict = mvnpdf(trainData, E_A_4_ur, CVM_A_4_ur) * A_x_apriori;
        A_5_aposteriori_predict = mvnpdf(trainData, E_A_5_ur, CVM_A_5_ur) * A_x_apriori;
        A_6_aposteriori_predict = mvnpdf(trainData, E_A_6_ur, CVM_A_6_ur) * A_x_apriori;
        A_7_aposteriori_predict = mvnpdf(trainData, E_A_7_ur, CVM_A_7_ur) * A_x_apriori;
        A_8_aposteriori_predict = mvnpdf(trainData, E_A_8_ur, CVM_A_8_ur) * A_x_apriori;
        A_9_aposteriori_predict = mvnpdf(trainData, E_A_9_ur, CVM_A_9_ur) * A_x_apriori;
    
        % L2 Norm der aposteriori Vorhersage
        A0_l2 = norm(A_0_aposteriori_predict);
        A1_l2 = norm(A_1_aposteriori_predict);
        A2_l2 = norm(A_2_aposteriori_predict);
        A3_l2 = norm(A_3_aposteriori_predict);
        A4_l2 = norm(A_4_aposteriori_predict);
        A5_l2 = norm(A_5_aposteriori_predict);
        A6_l2 = norm(A_6_aposteriori_predict);
        A7_l2 = norm(A_7_aposteriori_predict);
        A8_l2 = norm(A_8_aposteriori_predict);
        A9_l2 = norm(A_9_aposteriori_predict);
    
        % Bestimmung des Maximums (aposteriori Vorhersage)
        [maxValue, indexAtMaxValue] = max([A0_l2, A1_l2, A2_l2, A3_l2, A4_l2, A5_l2, A6_l2, A7_l2, A8_l2, A9_l2]);
    
        % Bayes Klassifikation (Welche aposteriori Vorhersage war die Groeßte?)
        if (maxValue == A0_l2)       % train 0 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),0];
            M_classify = vertcat(M_classify,tmpVector);
        elseif (maxValue == A1_l2)   % train 1 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),1];
            M_classify = vertcat(M_classify,tmpVector);
        elseif (maxValue == A2_l2)   % train 2 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),2];
            M_classify = vertcat(M_classify,tmpVector);
        elseif (maxValue == A3_l2)   % train 3 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),3];
            M_classify = vertcat(M_classify,tmpVector);
        elseif (maxValue == A4_l2)   % train 4 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),4];
            M_classify = vertcat(M_classify,tmpVector);
        elseif (maxValue == A5_l2)   % train 5 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),5];
            M_classify = vertcat(M_classify,tmpVector);
        elseif (maxValue == A6_l2)   % train 6 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),6];
            M_classify = vertcat(M_classify,tmpVector);
        elseif (maxValue == A7_l2)   % train 7 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),7];
            M_classify = vertcat(M_classify,tmpVector);
        elseif (maxValue == A8_l2)   % train 8 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),8];
            M_classify = vertcat(M_classify,tmpVector);
        else                         % train 9 predicted
            tmpVector = [B_ur(index,:),B(index,B_n),9];
            M_classify = vertcat(M_classify,tmpVector);
        end % end-if
    end % end-for_each
    
    M_classify_n = size(M_classify,2);
    M_classify_m = size(M_classify,1);

    % Konfusionsmatrix
    knownClass	 = M_classify(:, M_classify_n -1);
    predictedClass	 = M_classify(:, M_classify_n);
    disp(['Number of dimensions: ',num2str(dim)]);
    confusionmatrix	 = confusionmat(knownClass, predictedClass)

    % Klassifikationsguete
    corret_predicted = 0;
    for index = 1:M_classify_m
        if M_classify(index, M_classify_n -1) == M_classify(index, M_classify_n)
            corret_predicted = corret_predicted + 1;
        end
    end
    classification_quality = corret_predicted / M_classify_m
    
end % for dim