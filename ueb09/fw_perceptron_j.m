function [r] = fw_perceptron_j(x)

    % x: Spaltenvektor aus Input-Werten, multipliziert mit ihren weights
    % r: Spaltenvektor aus Output-Werten
    
    r = [];
    for i=1:size(x, 1)
        r = vertcat(r, (1 / (1+exp(-(x(i)))) ));
    end
end