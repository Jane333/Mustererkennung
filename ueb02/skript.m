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
  
%A_Training_clean = A_Training_unique_entries(1,:);
%k = 2
%while (k < A_n)
%  if (A_Training_unique_entries(k,1) == A_Training_unique_entries(k-1,1))
%    k = k+1;
%  else
%    A_Training_clean = vertcat(A_Training_clean,A_Training_unique_entries(k,:));
%    k = k+1
%  end
%end

% Testausgabe der neuen (bereinigten) Matrix:
%A_Training_clean; % <-- Ausgabe hier! (disabled with ;)


% 3. Problem: Mit K-NN-Algorithmus die Testdaten (B) mit Hilfe der Trainingsdaten klassifizieren
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Vorbereitung:
%A_dim = size(A_Training_clean);
%A_n = A_dim(1,1);
%A_m = A_dim(1,2);


% 3.1 Teilproblem: K = 1

C1 = [];

b = 1;
while (b < B_n +1)
  a = 1;
  while (a < A_n +1)
    if (a == A_n)
      tmpVector = [B_Testing(b,1),B_Testing(b,2),A_Training_unique_entries(A_n,2)];
      C1 = vertcat(C,tmpVector);
    elseif (B_Testing(b,1) > A_Training_unique_entries(a,1))
      a = a+1;
    else
      if ( B_Testing(b,1) - A_Training_unique_entries(a-1,1) < B_Testing(b,1) - A_Training_unique_entries(a,1) )
        tmpVector = [B_Testing(b,1),B_Testing(b,2),A_Training_unique_entries(a-1,2)];
      else
        tmpVector = [B_Testing(b,1),B_Testing(b,2),A_Training_unique_entries(a,2)];
      end
      C1 = vertcat(C,tmpVector);
      a = A_n +1;
    end
  end
  b = b+1;
end

% Ausgabe der Ergebnismatrix
% 1. Spalte: Gewicht, 2. Spalte: Futterklasse (Testdaten), 3. Spalte: Futterklasse (Trainingsdaten)
C1; % <-- Ausgabe hier! (disabled with ;)


% 3.2 Teilproblem: K = 3

C3 = [0,0,0];

b = 1;
while (b < B_n +1)
  a = 1;
  while (a < A_n +1)
    a = a +1;
  end
  b = b+1;
end

% Ausgabe der Ergebnismatrix
% 1. Spalte: Gewicht, 2. Spalte: Futterklasse (Testdaten), 3. Spalte: Futterklasse (Trainingsdaten)
C3 % <-- Ausgabe hier! (disable with ;)

% 3.3 Teilproblem: K = 5

C5 = [0,0,0];

b = 1;
while (b < B_n +1)
  a = 1;
  while (a < A_n +1)
    a = a+1;
  end
  b = b+1;
end

% Ausgabe der Ergebnismatrix
% 1. Spalte: Gewicht, 2. Spalte: Futterklasse (Testdaten), 3. Spalte: Futterklasse (Trainingsdaten)
C5; % <-- Ausgabe hier! (disabled with ;)


% 4. Problem: Erstellen der Konfusionsmatrizen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C1_bekannt = C1(:,2);
C1_vorhergesagt = C1(:,3);
Konfusionsmatrix_K1 = confusionmat(C1_bekannt,C1_vorhergesagt)

C3_bekannt = C3(:,2);
C3_vorhergesagt = C3(:,3);
Konfusionsmatrix_K3 = confusionmat(C3_bekannt,C3_vorhergesagt)

C5_bekannt = C5(:,2);
C5_vorhergesagt = C5(:,3);
Konfusionsmatrix_K5 = confusionmat(C5_bekannt,C5_vorhergesagt)


% 5. Problem: Klassifikationsgüte berechnen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% K=1
C1_korrekt = C1(C1(:,2)==C1(:,3),:);
correct = size(C1_korrekt);
all = size(C1);
Klassifikationsgute_K1 = correct(1,1) / all(1,1)

% K=3
C3_korrekt = C3(C3(:,2)==C3(:,3),:);
correct = size(C3_korrekt);
all = size(C3);
Klassifikationsgute_K3 = correct(1,1) / all(1,1)

% K=5
C5_korrekt = C5(C5(:,2)==C5(:,3),:);
correct = size(C5_korrekt);
all = size(C5);
Klassifikationsgute_K5 = correct(1,1) / all(1,1)

