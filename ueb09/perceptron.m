function [r] = perceptron(x,w)

    % x: Zeilenvektor aus input-Werten
    % w: Spaltenvektor aus weights-Werten
    
    % Schicht 1: 16    (Input-Layer)
    % Schicht 2: 2,4,8 (Hidden-Layer)
    % Schicht 3: 10    (Output-Layer)
    
    r = (1/(1+exp(-(x*w))));
    
end