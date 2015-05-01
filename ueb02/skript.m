% Trainingsdaten (A) und Testdaten (B) laden
A = load('chickwts_training.csv');
B = load('chickwts_testing.csv');

% Trainingsdaten aufbereiten
A_Training = A(:,2:3);
A_Training = sortrows(A_Training,1);
A_dim = size(A_Training);
A_n = A_dim(1,1);
A_m = A_dim(1,2);
A_Training_unique_entries = unique(A_Training,'rows');
A_Training_clean = A_Training_unique_entries(1,:);

for k = 2:(A_n-1)
  if ~(A_Training_unique_entries(k,1) == A_Training_unique_entries(k+1,1))
    vertcat(A_Training_clean,A_Training_unique_entries(k,:));
  end
end
A_Training_clean

% Testingdaten aufbereiten
B_Testing = B(:,2:3);
B_Testing = sortrows(B_Testing,1);
B_dim = size(B_Testing);
B_n = B_dim(1,1);
B_m = B_dim(1,2);


