% AUFGABE 1

function y = classifier() % h ist das zu klassifizierende Huhn
	
	A = load('chickwts_training.csv'); % Spalte 1 = Huhn-ID, Spalte 2 = Gewicht, Spalte 3 = Futterklasse
	A_Sorted = sortrows(A,2); % nach Gewichten sortieren
	A_Training = A(:,2:3);
	U = unique(A_Training,'rows');
	zeilen = size(U,1);
	B = load('chickwts_testing.csv');
	B_Testing = B(:,2:3);
	
	for i = 1:3
		if i == 1
			k = 1 % dies ist *das* k, as in, k-nearest neighbor
		elseif i == 2
			k = 3
		elseif i == 3
			k = 5
		end
	
		dists = zeros(2*k - 1, 2); %	(2*k-1)x2-Matrix mit den 5 Kandidat-Nachbarn um unser Eingabehuhn h herum. 3 von diesen 5 Nachbarn sind die 3 nearest neighbors unseres Huhns. 1. Spalte enthaelt die Distanzen, 2. Spalte enthaelt die Klassen dieser Nachbarn.
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
end
