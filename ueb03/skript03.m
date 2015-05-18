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
X = A(:,1:A_n -1); % alle Daten ausser der Zuglinie
x = min(X):max(X);
B_n   = size(B,2);
B_m   = size(B,1);

% Aufgabe 1 (3 Punkte)

% A 1.1: multivariate (mehrdimensionale) Normalverteilung
%        (Erwartungswert, Kovarianzmatrix) berechnen

% Erwartungswert fuer jede Koordinate fuer jeden Zug (0 bis 9)
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

% Kovarianzmatrix fuer jeden Zug (0 bis 9)
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

% Multivariante PDF generieren fuer jeden Zug (0 bis 9)
% wir geben hier kein Intervall an, weil die pdf hochdimensional ist 
% und nicht nur fuer einen bestimmten Bereich berechnet werden soll
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

M_classify;
%cp = classperf(B,c)

% Konfusionsmatrix
knownClass = M_classify(:, B_n);
predictedClass = M_classify(:, B_n +1);
confusionmat(knownClass, predictedClass)

% Klassifikationsguete
M_m = size(M_classify, 1);
corret_predicted = 0;
for index = 1:M_m
    if M_classify(index, B_n) == M_classify(index, B_n +1)
        corret_predicted = corret_predicted + 1;
    end
end
classification_quality = corret_predicted / M_m


%%%%%%%%%%%%%%%%%%%%%%%%%%  Aufgabe 2 (4 Punkte)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A 2.1: Erste Hauptkomponente der Trainingsdaten angeben
M_pca = princomp(zscore(B(:,1:B_n -1))); % pca with standardized variables
M_pca_fst = M_pca(:,1); % ?

% A 2.2: Dimensionsreduzierung mittels PCA,
%        Testdaten klassifizieren mit Bayes Klassifikator
%        (wie in Aufgabe 1)
%        Klassifikationsguete fuer alle Dimensionen angeben


%%%%%%%%%%%%%%%%%%%%%%%%%%%  Aufgabe 3 (3 Punkte)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A 3.1: k-means auf die Daten clusters.txt anwenden,
%        k-means soll selbst implementiert werden!

k = 3;
numIterations = 5

mean1 = C(1,:) % mean1, selected randomly
mean2 = C(2,:) % mean2, selected randomly
mean3 = C(3,:) % mean3, selected randomly
mean1_elems = [] % elements belonging to mean1
mean2_elems = [] % elements belonging to mean2
mean3_elems = [] % elements belonging to mean3
plotArray = []

for iter=1:numIterations
    mean1_elems = []
    mean2_elems = []
    mean3_elems = []
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
    mean1 = [mean(mean1_elems(:,1)), mean(mean1_elems(:,2))]
    mean2 = [mean(mean2_elems(:,1)), mean(mean2_elems(:,2))]
    mean3 = [mean(mean3_elems(:,1)), mean(mean3_elems(:,2))]
    
    % Visualisierung der Clusterzentren
    plotOfIteration = 5 % which iteration do we want to see a plot for?
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
