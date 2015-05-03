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

% alle doppelten Einträge aus der Matrix A entfernen
A_Training_unique_entries = unique(A_Training,'rows');

% Für ein besseres Handling: die Dimensionen der Matrizen 
A_dim = size(A_Training_unique_entries);
B_dim = size(B_Testing);
A_n = A_dim(1,1);
A_m = A_dim(1,2);
B_n = B_dim(1,1);
B_m = B_dim(1,2);


% 2. Problem: Trainingsdaten aufbereiten
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wir entfernen alle Einträge von Hühnern mit dem gleichen Gewicht.
% Falls es mehr als ein Huhn gibt, nehmen wir das letzte Huhn.
  
A_Training_clean = A_Training_unique_entries(1,:);
k = 2;
while (k < A_n)
  if (A_Training_unique_entries(k,1) == A_Training_unique_entries(k-1,1))
    k = k+1;
  else
    A_Training_clean = vertcat(A_Training_clean,A_Training_unique_entries(k,:));
    k = k+1;
  end
end

% Testausgabe der neuen (bereinigten) Matrix:
%  A_Training_clean;


% 3. Problem: Mit K-NN-Algorithmus die Testdaten (B) mit Hilfe der Trainingsdaten klassifizieren
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Vorbereitung:
A_dim = size(A_Training_clean);
A_n = A_dim(1,1);
A_m = A_dim(1,2);


% 3.1 Teilproblem: K = 1

C = [];

b = 1;
while (b < B_n +1)
  a = 1;
  while (a < A_n +1)
    if (a == A_n)
      tmpVector = [B_Testing(b,1),B_Testing(b,2),A_Training_clean(A_n,2)];
    elseif (B_Testing(b,1) > A_Training_clean(a,1))
      a = a+1;
    else
      if ( B_Testing(b,1) - A_Training_clean(a-1,1) < B_Testing(b,1) - A_Training_clean(a,1) )
        tmpVector = [B_Testing(b,1),B_Testing(b,2),A_Training_clean(a-1,2)];
      else
        tmpVector = [B_Testing(b,1),B_Testing(b,2),A_Training_clean(a,2)];
      end
      C = vertcat(C,tmpVector);
      a = A_n +1;
    end
  end
  b = b+1;
end

% Aufgabe der Ergebnismatrix
% 1. Spalte: Gewicht, 2. Spalte: Futterklasse (Testdaten), 3. Spalte: Futterklasse (Trainingsdaten)
%  C
% Konfusionsmatrix fuer K = 1
knownClass = C(:, 2);
predictedClass = C(:, 3);
confusionMatrix = confusionmat(knownClass, predictedClass)

% Klassifikationsguete fuer K = 1
alle = size(C, 1);
korrekt_vorhergesagt = 0;
for z = 1:alle
  if C(z, 2) == C(z, 3)
    korrekt_vorhergesagt = korrekt_vorhergesagt + 1;
  end
end
Klassifikationsguete = korrekt_vorhergesagt / alle


% 3.2 Teilproblem: K = 3


% 3.3 Teilproblem: K = 5



% 4. Problem: Erstellen der Konfusionsmatrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 5. Problem: Klassifikationsgüte berechnen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


