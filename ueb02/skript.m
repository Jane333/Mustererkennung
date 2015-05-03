% Trainingsdaten (A) und Testdaten (B) laden
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = load('chickwts_training.csv');
B = load('chickwts_testing.csv');


% 1. Problem: Daten vorbereiten
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Aus den Daten nur Spalte 2 und 3 betrachten und neu erzeugte Matrix sortieren:
A_Training = A(:,2:3);
A_Training = sortrows(A_Training,1);
B_Testing = B(:,2:3);
B_Testing = sortrows(B_Testing,1);

% Für ein besseres Handling: die Dimensionen der Matrizen 
A_dim = size(A_Training_unique_entries);
B_dim = size(B_Testing);
A_n = A_dim(1,1);
A_m = A_dim(1,2);
B_n = B_dim(1,1);
B_m = B_dim(1,2);

% alle doppelten Einträge aus der Matrix A entfernen
A_Training_unique_entries = unique(A_Training,'rows');


% 2. Problem: Trainingsdaten aufbereiten
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wir entfernen alle Einträge von Hühnern mit dem gleichen Gewicht.
% Falls es mehr als ein Huhn gibt, nehmen wir das letzte Huhn.
  
A_Training_clean = A_Training_unique_entries(1,:);
k = 2
while (k < A_n)
  if (A_Training_unique_entries(k,1) == A_Training_unique_entries(k-1,1))
    k = k+1;
  else
    A_Training_clean = vertcat(A_Training_clean,A_Training_unique_entries(k,:));
    k = k+1
  end
end

% Testausgabe der neuen (bereinigten) Matrix:
A_Training_clean


% 3. Problem: Mit K-NN-Algorithmus die Testdaten (B) mit Hilfe der Trainingsdaten klassifizieren
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3.1 Teilproblem: K = 1


% 3.2 Teilproblem: K = 3


% 3.3 Teilproblem: K = 5



% 4. Problem: Erstellen der Konfusionsmatrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 5. Problem: Klassifikationsgüte berechnen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


