function [result] = error_calculation(netout,labels)

    % netout: network output
    % labels: labels
    
    result = [];
    for index = 1:length(labels)
        result = vertcat(result,netout(index)-labels(index));
    end
end