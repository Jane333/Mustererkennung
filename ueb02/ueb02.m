% AUFGABE 1

function y = classifier() % h ist das zu klassifizierende Huhn
  k = 1; % dies ist *das* k, as in, k-nearest neighbor
  A = load('chickwts_training.csv'); % Spalte 1 = Huhn-ID, Spalte 2 = Gewicht, Spalte 3 = Futterklasse
  B = sortrows(A,2); % nach Gewichten sortieren
  h = 130;
  zeilen = size(B,1);
  dists = zeros(2*k - 1, 2); %  (2*k-1)x2-Matrix mit den 5 Kandidat-Nachbarn um unser Eingabehuhn h herum. 3 von diesen 5 Nachbarn sind die 3 nearest neighbors unseres Huhns. 1. Spalte enthaelt die Distanzen, 2. Spalte enthaelt die Klassen dieser Nachbarn.
  dists = dists - 1;
  
  % Huhn 'treffer' finden, das unserem Huhn h am naechsten ist:
  treffer = -1;
  trefferZeile = -1; % Zeile, in welcher wir das 'treffer'-Huhn gefunden haben
  zeilen = size(B,1);
  if h < B(1,2)
    treffer = B(1,3);
    trefferZeile = 1;
  elseif h > B(zeilen,2)
    treffer = B(zeilen,3);
    trefferZeile = zeilen;
  else
  
    for z = 1:zeilen
      if h == B(z, 2)
        treffer = B(z, 3);
        trefferZeile = z;
        break
      elseif h < B(z,2)
        dist1 = abs(B(z-1, 2) - h);
        dist2 = abs(B(z, 2) - h);
        if dist1 <= dist2
          treffer = B(z-1,3);
          trefferZeile = z;
          break
        else
          treffer = B(z,3);
          trefferZeile = z;
          break
        end
      end
    end  
  end
  % Wir haben unseren 'treffer' gefunden. Nun entscheiden wir, welche Huehner um ihn herum unsere k naechsten Nachbarn sind.
  if k == 1
    y = treffer;
  elseif k > 1
    offset = 0;
    if trefferZeile < k
      offset = k - trefferZeile; % 3-1 = 2
    elseif trefferZeile > zeilen - k + 1
      offset = zeilen - k + 1 - trefferZeile; % 10-3+1-10 = -2
    end
    dists = B(trefferZeile-2+offset:trefferZeile+2+offset, 2:3); % 1-2+2 = 1
    dists = [abs((dists(:,1)) - h), dists(:,2)] % Gewichte ersetzen durch Distanzen der Gewichte zu h. TODO: ich versuche hier, jeden Eintrag der 1. Spalte von dists zu ersetzen mit abs(Eintrag - h), aber keine Ahnung, ob dies die richtige Syntax ist.
    dists = sortrows(dists,1) % nach Gewicht-Distanzen sortieren
    k_neighbors_classes = dists(1:k, 2) % die Klassen der k Nachbarn mit den kleinsten Gewicht-Distanzen holen
    most_frequent_neighbor_class = mode(k_neighbors_classes) % findet die haeufigste Klasse in k_neighbors_classes
    y = most_frequent_neighbor_class
  end
end
