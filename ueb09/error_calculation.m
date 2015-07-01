function [result] = error_calculation(layer2,labels)

    % netout: network output
    % labels: labels
    
    result = [];
    for index = 1:length(labels)
        result = vertcat(result,layer2(index)-labels(index));
    end
end