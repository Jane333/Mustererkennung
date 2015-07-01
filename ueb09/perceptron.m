function [R] = perceptron(I,W)
    
    % Schicht 1: 16    (Input-Layer)
    % Schicht 2: 2,4,8 (Hidden-Layer)
    % Schicht 3: 10    (Output-Layer)
    
    R = (1/(1+exp(-(I*W'))));
    
end