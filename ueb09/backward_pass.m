function [W1_new,W2_new] = backward_pass(layer0,layer1,layer2,W1,W2,error)

    % netout: forward pass result vector
    % W1:     weight matrix for layer 1
    % W2:     weight matrix for layer 2
    % error:  error vector
    
    W1_ = W1(1:(length(W1)-1),:)
    W2_ = W2(1:(length(W2)-1),:)
    
    layer1
    layer2
    
    t1     = layer1*W1_;
    W1_new = 1/1+exp(-t1);
    D1     = zeros(length(W1_new));
    for d1 = 1:length(W1_new)
        D1(d1,d1) = W1_new(d1);
    end
    D1
    
    %W2_new_transpose = -alpha*D2*error*layer2
    %W2_new = W2_new_transpose'
    %W1_new_transpose = -alpha*D1*W2*D2*error*layer0
    
    W1_new = W1;
    W2_new = W2;
end