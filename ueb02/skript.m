coder.extrinsic('disp')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% AUFGABE 1

disp('Aufgabe 1')

% Spalte 1 = Huhn-ID, Spalte 2 = Gewicht, Spalte 3 = Futterklasse
A = load('chickwts_training.csv');
A_Sorted = sortrows(A,2); % nach Gewichten sortieren
A_Training = A(:,2:3);
U = unique(A_Training,'rows');
zeilen = size(U,1);

% Spalte 1 = Huhn-ID, Spalte 2 = Gewicht, Spalte 3 = Futterklasse
B = load('chickwts_testing.csv');
B_Testing = B(:,2:3);

for k = [1,3,5]

  dists = zeros(2*k - 1, 2);
  % (2*k-1)x2-Matrix mit den 5 Kandidat-Nachbarn um unser Eingabehuhn h herum.
  % 3 von diesen 5 Nachbarn sind die 3 nearest neighbors unseres Huhns.
  % 1. Spalte enthaelt die Distanzen, 2. Spalte enthaelt die Klassen dieser Nachbarn.
  dists = dists - 1; % weil ich eine -1 als Initialwert schoener finde als ne 0
		
  C = []; % Ergebnismatrix
 
  for zeilenindex_testdaten = 1:size(A_Training,1)
    h = A_Training(zeilenindex_testdaten, 1); % unser Testhuhn
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
    
    if k == 1
      most_frequent_neighbor_class = treffer;
    elseif k > 1
      offset = 0;
      if trefferZeile < k
        offset = k - trefferZeile; % 3-1 = 2
      elseif trefferZeile > zeilen - k + 1
        offset = zeilen - k + 1 - trefferZeile; % 10-3+1-10 = -2
      end
      % aus U ein Fenster der Laenge 2*k-1 um 'treffer' herum ausschneiden
      dists = U(trefferZeile-(k-1)+offset:trefferZeile+(k-1)+offset, :);
      % Gewichte ersetzen durch Distanzen der Gewichte zu h.
      dists = [abs((dists(:,1)) - h), dists(:,2)]; 
      % nach Gewicht-Distanzen sortieren
      dists = sortrows(dists,1);
      % die Klassen der k Nachbarn mit den kleinsten Gewicht-Distanzen holen
      k_neighbors_classes = dists(1:k, 2);
      % findet die haeufigste Klasse in k_neighbors_classes
      most_frequent_neighbor_class = mode(k_neighbors_classes);
    end
    % Ergebnismatrix C: 1. Spalte: Gewicht, 2. Spalte: Futterklasse (Testdaten), 3. Spalte: Futterklasse (Trainingsdaten)
    tmpVector = [A_Training(zeilenindex_testdaten, 1), A_Training(zeilenindex_testdaten, 2), most_frequent_neighbor_class];
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Aufgabe 2

disp('Aufgabe 2')

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

% Varianzen berechnen
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

% PDF berechnen
FM0_pdf = pdf('Normal',A_Training(:,1), FM0_mean,FM0_std);
FM1_pdf = pdf('Normal',A_Training(:,1), FM0_mean,FM1_std);
FM2_pdf = pdf('Normal',A_Training(:,1), FM0_mean,FM2_std);
FM3_pdf = pdf('Normal',A_Training(:,1), FM0_mean,FM3_std);
FM4_pdf = pdf('Normal',A_Training(:,1), FM0_mean,FM4_std);
FM5_pdf = pdf('Normal',A_Training(:,1), FM0_mean,FM5_std);

% PDFs plotten (geht noch nicht... :-( )
plot(min(FM0_matrix),(max(FM0_matrix)-min(FM0_matrix))/100,max(FM0_matrix),FM0_pdf);
plot(min(FM1_matrix),(max(FM1_matrix)-min(FM1_matrix))/100,max(FM1_matrix),FM0_pdf);
plot(min(FM2_matrix),(max(FM2_matrix)-min(FM2_matrix))/100,max(FM2_matrix),FM0_pdf);
plot(min(FM3_matrix),(max(FM3_matrix)-min(FM3_matrix))/100,max(FM3_matrix),FM0_pdf);
plot(min(FM4_matrix),(max(FM4_matrix)-min(FM4_matrix))/100,max(FM4_matrix),FM0_pdf);
plot(min(FM5_matrix),(max(FM5_matrix)-min(FM5_matrix))/100,max(FM5_matrix),FM0_pdf);
