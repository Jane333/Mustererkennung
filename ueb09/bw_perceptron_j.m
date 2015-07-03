function [D_out] = bw_perceptron_j(D, x)

    % x: Spaltenvektor aus Input-Werten, multipliziert mit ihren weights
    % D: Einheitsmatrix
    % D_out: Matrix mit Ableitungen der Sigmoid-Funktion auf der Hauptdiagonale
    
    D_out = [];
    for i=1:size(x, 1)
        sigmoid = (1 / (1+exp(-(x(i)))) );
        sigmoid_derivative = sigmoid * (1 - sigmoid);
        D_out = vertcat(D_out, D(i,:) .* sigmoid_derivative);
    end
end