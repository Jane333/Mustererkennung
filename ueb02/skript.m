coder.extrinsic('disp')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% AUFGABE 1 - Code. Loesung siehe weiter unten
% Notiz: Bei der Loesung dieser Aufgaben haben wir mit Daten gearbeitet, von welchen alle Duplikate entfernt wurden (ein Duplikat ist ein Huhn, welches die gleiche Futterklasse UND das gleiche Gewicht hat wie ein anderes  Huhn im Datensatz.)

disp('Aufgabe 1')

% Spalte 1 = Huhn-ID, Spalte 2 = Gewicht, Spalte 3 = Futterklasse

A = load('chickwts_training.csv');
A_Sorted = sortrows(A,2); % nach Gewichten sortieren
A_Training = A(:,2:3);
U = unique(A_Training,'rows');
zeilen = size(U,1);
B = load('chickwts_testing.csv');
B_Testing = B(:,2:3);

alle_k = [1 3 5];
for k=alle_k
  fprintf('k = %i\n',k)
  dists = zeros(2*k - 1, 2); %  (2*k-1)x2-Matrix mit den 5 Kandidat-Nachbarn 
  % um unser Eingabehuhn h herum. 3 von diesen 5 Nachbarn sind die 3 nearest 
  % neighbors unseres Huhns. 1. Spalte enthaelt die Distanzen, 2. Spalte enthaelt
  % die Klassen dieser Nachbarn.
  
  C = []; % Ergebnismatrix
  for zeilenindex_testdaten = 1:size(B_Testing,1)
    h = B_Testing(zeilenindex_testdaten, 1); % unser Testhuhn
    % Huhn 'treffer' finden, das unserem Huhn h am naechsten ist:
    treffer = -1;
    trefferZeile = -1; % Zeile, in welcher wir das 'treffer'-Huhn gefunden haben
    if h < U(1,1)
      treffer = U(1,2);
      trefferZeile = 1;
    elseif h > U(zeilen,1)
      treffer = U(zeilen,2);
      trefferZeile = zeilen;
    else
    
      for z = 1:zeilen
        if h == U(z, 1)
          treffer = U(z, 2);
          trefferZeile = z;
          break
        elseif h < U(z,1)
          dist1 = abs(U(z-1, 1) - h);
          dist2 = abs(U(z, 1) - h);
          if dist1 <= dist2
            treffer = U(z-1,2);
            trefferZeile = z;
            break
          else
            treffer = U(z,2);
            trefferZeile = z;
            break
          end
        end
      end  
    end
    % Wir haben unseren 'treffer' gefunden. Nun entscheiden wir, welche Huehner um ihn herum unsere k naechsten Nachbarn sind.
    if k == 1
      most_frequent_neighbor_class = treffer;
    elseif k > 1
      offset = 0;
      if trefferZeile < k
        offset = k - trefferZeile; % 3-1 = 2
      elseif trefferZeile > zeilen - k + 1
        offset = zeilen - k + 1 - trefferZeile; % 10-3+1-10 = -2
      end
      dists = U(trefferZeile-(k-1)+offset:trefferZeile+(k-1)+offset, :); % aus U ein Fenster der Laenge 2*k-1 um 'treffer' herum ausschneiden
      dists = [abs((dists(:,1)) - h), dists(:,2)]; % Gewichte ersetzen durch Distanzen der Gewichte zu h.
      dists = sortrows(dists,1); % nach Gewicht-Distanzen sortieren
      k_neighbors_classes = dists(1:k, 2); % die Klassen der k Nachbarn mit den kleinsten Gewicht-Distanzen holen
      most_frequent_neighbor_class = mode(k_neighbors_classes); % findet die haeufigste Klasse in k_neighbors_classes
    end
    
    % Ergebnismatrix C: 1. Spalte: Gewicht, 2. Spalte: Futterklasse (Testdaten), 3. Spalte: Futterklasse (Trainingsdaten)
    tmpVector = [B_Testing(zeilenindex_testdaten, 1), B_Testing(zeilenindex_testdaten, 2), most_frequent_neighbor_class];
      C = vertcat(C,tmpVector);
      
  end % end of for each test chicken
  
  % Konfusionsmatrix
  knownClass = C(:, 2);
  predictedClass = C(:, 3);
  confusionMatrix = confusionmat(knownClass, predictedClass)

  % Klassifikationsguete
  alle = size(C, 1);
  korrekt_vorhergesagt = 0;
  for z = 1:alle
    if C(z, 2) == C(z, 3)
      korrekt_vorhergesagt = korrekt_vorhergesagt + 1;
    end
  end
  Klassifikationsguete = korrekt_vorhergesagt / alle
end % end of for k in 1, 3, 5


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
% AUFGABE 1 - Loesung
%  
%  k = 1
%  
%  confusionMatrix =
%  
%       3     3     1     0     3     0
%       5     6     1     0     0     0
%       2     2     4     3     3     0
%       0     1     1     6     2     2
%       3     1     1     4     1     1
%       0     3     3     3     0     3
%  
%  
%  Klassifikationsguete = 0.3239
%  
%  
%  k = 3
%  
%  confusionMatrix =
%  
%       7     0     2     0     1     0
%       6     4     0     0     2     0
%       3     0     6     2     3     0
%       1     1     2     7     0     1
%       4     1     1     4     1     0
%       0     3     3     3     0     3
%  
%  Klassifikationsguete = 0.3944
%  
%  
%  k = 5
%  
%  confusionMatrix =
%  
%       7     0     2     0     1     0
%       6     3     0     1     2     0
%       2     1     7     1     3     0
%       0     0     3     7     1     1
%       2     2     1     5     1     0
%       0     1     3     3     0     5
%  
%  Klassifikationsguete = 0.4225


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Aufgabe 2

disp('Aufgabe 2')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Teilaufgabe a)                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Berechnungen fuer die 6 Futtermittel FM0..FM5:

FM0_matrix = A_Training(A_Training(:,2)==0,1);
FM1_matrix = A_Training(A_Training(:,2)==1,1);
FM2_matrix = A_Training(A_Training(:,2)==2,1);
FM3_matrix = A_Training(A_Training(:,2)==3,1);
FM4_matrix = A_Training(A_Training(:,2)==4,1);
FM5_matrix = A_Training(A_Training(:,2)==5,1);

% Erwartungswert/Mittelwert berechnen
FM0_mean = mean(FM0_matrix);
FM1_mean = mean(FM1_matrix);
FM2_mean = mean(FM2_matrix);
FM3_mean = mean(FM3_matrix);
FM4_mean = mean(FM4_matrix);
FM5_mean = mean(FM5_matrix);

% Varianz
FM0_var = var(FM0_matrix);
FM1_var = var(FM1_matrix);
FM2_var = var(FM2_matrix);
FM3_var = var(FM3_matrix);
FM4_var = var(FM4_matrix);
FM5_var = var(FM5_matrix);

% Standardabweichungen
FM0_std = std(FM0_matrix);
FM1_std = std(FM1_matrix);
FM2_std = std(FM2_matrix);
FM3_std = std(FM3_matrix);
FM4_std = std(FM4_matrix);
FM5_std = std(FM5_matrix);

% A Priori Wahrscheinlickeit berechnen
FM0_apriori = length(FM0_matrix) / length(A_Training)
FM1_apriori = length(FM1_matrix) / length(A_Training)
FM2_apriori = length(FM2_matrix) / length(A_Training)
FM3_apriori = length(FM3_matrix) / length(A_Training)
FM4_apriori = length(FM4_matrix) / length(A_Training)
FM5_apriori = length(FM5_matrix) / length(A_Training)

%%%%% Ergebnisse zu Teilaufgabe a) %%%%%
% 
% FM0_mean = 165.7200
% FM1_mean = 217.9167
% FM2_mean = 244.6429
% FM3_mean = 325.6000
% FM4_mean = 281.8182
% FM5_mean = 326.9333
% 
% FM0_var = 2.0801e+03
% FM1_var = 3.8649e+03
% FM2_var = 3.6319e+03
% FM3_var = 3.2483e+03
% FM4_var = 5.3364e+03
% FM5_var = 4.8296e+03
% 
% FM0_std = 45.6079
% FM1_std = 62.1682
% FM2_std = 60.2656
% FM3_std = 56.9937
% FM4_std = 73.0505
% FM5_std = 69.4952
% 
% FM0_apriori = 0.1408
% FM1_apriori = 0.1690
% FM2_apriori = 0.1972
% FM3_apriori = 0.1690
% FM4_apriori = 0.1549
% FM5_apriori = 0.1690

% PDFs plotten (in einer Grafik)
X = A_Training(:,1);
x = min(X):max(X); % Abschnitt auf der x-Achse, der geplottet werden soll

% PDFs berechnen
FM0_pdf = pdf('Normal',x,FM0_mean, FM0_std); % pdf(Art von Verteilung, Abschnitt auf x-Achse, mean, std)
FM0_pdf
FM1_pdf = pdf('Normal',x,FM1_mean, FM1_std);
FM2_pdf = pdf('Normal',x,FM2_mean, FM2_std);
FM3_pdf = pdf('Normal',x,FM3_mean, FM3_std);
FM4_pdf = pdf('Normal',x,FM4_mean, FM4_std);
FM5_pdf = pdf('Normal',x,FM5_mean, FM5_std);


%P1 = plot(x,FM0_pdf,x,FM1_pdf,x,FM2_pdf,x,FM3_pdf,x,FM4_pdf,x,FM5_pdf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Teilaufgabe b)                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FM0_aposteriori = FM0_pdf * FM0_apriori;
FM1_aposteriori = FM1_pdf * FM1_apriori;
FM2_aposteriori = FM2_pdf * FM2_apriori;
FM3_aposteriori = FM3_pdf * FM3_apriori;
FM4_aposteriori = FM4_pdf * FM4_apriori;
FM5_aposteriori = FM5_pdf * FM5_apriori;

% PDFs plotten (in einer Grafik)

P2 = plot(x,FM0_aposteriori,x,FM1_aposteriori,x,FM2_aposteriori,x,FM3_aposteriori,x,FM4_aposteriori,x,FM5_aposteriori);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Teilaufgabe c)                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Idee:
% Für jedes Huhn aus B_Testing multipliziere sein Gewicht mit FMX_aposteriori.
% Wenn FMX_aposeriori > FMY_aposteriori, dann wähle FMY als Vorhersage.
D = [];
for huhnIndex = 1:size(B_Testing,1)
  h = B_Testing(huhnIndex, 1);
  fm0pre = pdf('Normal', h, FM0_mean, FM0_std) * FM0_apriori;
  fm1pre = pdf('Normal', h, FM1_mean, FM0_std) * FM1_apriori;
  fm2pre = pdf('Normal', h, FM2_mean, FM0_std) * FM2_apriori;
  fm3pre = pdf('Normal', h, FM3_mean, FM0_std) * FM3_apriori;
  fm4pre = pdf('Normal', h, FM4_mean, FM0_std) * FM4_apriori;
  fm5pre = pdf('Normal', h, FM5_mean, FM0_std) * FM5_apriori;
  
  [maxValue, indexAtMaxValue] = max([fm0pre,fm1pre,fm2pre,fm3pre,fm4pre,fm5pre]);
  
  if (maxValue == fm0pre)
    tmpVector = [B_Testing(huhnIndex,1),B_Testing(huhnIndex,2),0];
    D = vertcat(D,tmpVector);
  elseif (maxValue == fm1pre)
    tmpVector = [B_Testing(huhnIndex,1),B_Testing(huhnIndex,2),1];
    D = vertcat(D,tmpVector);
  elseif (maxValue == fm2pre)
    tmpVector = [B_Testing(huhnIndex,1),B_Testing(huhnIndex,2),2];
    D = vertcat(D,tmpVector);  
  elseif (maxValue == fm3pre)
    tmpVector = [B_Testing(huhnIndex,1),B_Testing(huhnIndex,2),3];
    D = vertcat(D,tmpVector);
  elseif (maxValue == fm4pre)
    tmpVector = [B_Testing(huhnIndex,1),B_Testing(huhnIndex,2),4];
    D = vertcat(D,tmpVector);
  else
    tmpVector = [B_Testing(huhnIndex,1),B_Testing(huhnIndex,2),5];
    D = vertcat(D,tmpVector);
  end
end % end of for each h

% Konfusionsmatrix
  knownClass = D(:, 2);
  predictedClass = D(:, 3);
  confusionMatrix = confusionmat(knownClass, predictedClass)

  % Klassifikationsguete
  alle = size(D, 1);
  korrekt_vorhergesagt = 0;
  for z = 1:alle
    if D(z, 2) == D(z, 3)
      korrekt_vorhergesagt = korrekt_vorhergesagt + 1;
    end
  end
  Klassifikationsguete = korrekt_vorhergesagt / alle
  
% confusionMatrix =
%
%     8     1     1     0     0     0
%     4     2     5     1     0     0
%     2     2     7     1     0     2
%     0     0     1     3     2     6
%     1     1     4     3     0     2
%     0     1     2     1     1     7
%
% Klassifikationsguete = 0.3803