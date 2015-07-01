function [W1_new,W2_new] = backward_pass(netout,W1,W2,error)

    % netout: forward pass result vector
    % W1:     weight matrix for layer 1
    % W2:     weight matrix for layer 2
    % error:  error vector
    
    W1_ = W1(1:(length(W1)-1),:);
    W2_ = W2(1:(length(W2)-1),:);
    
    W1_new = W1;
    W2_new = W2;
end