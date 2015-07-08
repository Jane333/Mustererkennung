% clean up
clear all
close all
clc

% load data
Data   = load('pendigits-training.txt');

% prepare data for network
LData1 = Data(1:60,:);
AData1 = horzcat(Data(1:60,1:16),ones(60,1));
Label1 = Data(1:60,17);

LData2 = Data(61:90,:);
AData2 = horzcat(Data(61:90,1:16),ones(30,1));
Label2 = Data(61:90,17);

%%% NETWORK DEFINITION %%%

% 16 Nodes in Layer0 - Inputlayer
% 16 Nodes in Layer1 - Hiddenlayer
% 10 Nodes in Layer2 - Outputlayer

% weights
W0 = ones(17,16)*(-0.5);
W1 = ones(17,10)*(-0.5);

% learning rate
alpha = 1;


% training
for i = 1:60
    D = AData1(i,:);
    L = Label1(i,:);
    
    % forward pass - layer 1
    t = D * W0;
    perceptron01_layer1 = 1 / (1 + exp(-t(:,1)));
    perceptron02_layer1 = 1 / (1 + exp(-t(:,2)));
    perceptron03_layer1 = 1 / (1 + exp(-t(:,3)));
    perceptron04_layer1 = 1 / (1 + exp(-t(:,4)));
    perceptron05_layer1 = 1 / (1 + exp(-t(:,5)));
    perceptron06_layer1 = 1 / (1 + exp(-t(:,6)));
    perceptron07_layer1 = 1 / (1 + exp(-t(:,7)));
    perceptron08_layer1 = 1 / (1 + exp(-t(:,8)));
    perceptron09_layer1 = 1 / (1 + exp(-t(:,9)));
    perceptron10_layer1 = 1 / (1 + exp(-t(:,10)));
    perceptron11_layer1 = 1 / (1 + exp(-t(:,11)));
    perceptron12_layer1 = 1 / (1 + exp(-t(:,12)));
    perceptron13_layer1 = 1 / (1 + exp(-t(:,13)));
    perceptron14_layer1 = 1 / (1 + exp(-t(:,14)));
    perceptron15_layer1 = 1 / (1 + exp(-t(:,15)));
    perceptron16_layer1 = 1 / (1 + exp(-t(:,16)));
    
    out_layer1 = [perceptron01_layer1;perceptron02_layer1;
                  perceptron03_layer1;perceptron04_layer1;
                  perceptron05_layer1;perceptron06_layer1;
                  perceptron07_layer1;perceptron08_layer1;
                  perceptron09_layer1;perceptron10_layer1;
                  perceptron11_layer1;perceptron12_layer1;
                  perceptron13_layer1;perceptron14_layer1;
                  perceptron15_layer1;perceptron16_layer1]';
    
    % forward pass - layer 2
    t = [out_layer1, 1]*W1;
    perceptron01_layer2 = 1 / (1 + exp(-t(:,1)));
    perceptron02_layer2 = 1 / (1 + exp(-t(:,2)));
    perceptron03_layer2 = 1 / (1 + exp(-t(:,3)));
    perceptron04_layer2 = 1 / (1 + exp(-t(:,4)));
    perceptron05_layer2 = 1 / (1 + exp(-t(:,5)));
    perceptron06_layer2 = 1 / (1 + exp(-t(:,6)));
    perceptron07_layer2 = 1 / (1 + exp(-t(:,7)));
    perceptron08_layer2 = 1 / (1 + exp(-t(:,8)));
    perceptron09_layer2 = 1 / (1 + exp(-t(:,9)));
    perceptron10_layer2 = 1 / (1 + exp(-t(:,10)));
    
    out_layer2 = [perceptron01_layer2;perceptron02_layer2;
                  perceptron03_layer2;perceptron04_layer2;
                  perceptron05_layer2;perceptron06_layer2;
                  perceptron07_layer2;perceptron08_layer2;
                  perceptron09_layer2;perceptron10_layer2]';
    
    % error calculation
    LV = zeros(1,10);
    for j = 1:10
        if L == j
            LV(1,j) = 1;
        end
    end
    error = 0.5*((out_layer2 - LV)'*(out_layer2 - LV));
    
    % backward pass - layer 1
    t = D * W0;
    
    s101 = (1 / (1+exp(-t(:,1)))) *(1-(1 / (1+exp(-t(:,1)))));
    s102 = (1 / (1+exp(-t(:,2)))) *(1-(1 / (1+exp(-t(:,2)))));
    s103 = (1 / (1+exp(-t(:,3)))) *(1-(1 / (1+exp(-t(:,3)))));
    s104 = (1 / (1+exp(-t(:,4)))) *(1-(1 / (1+exp(-t(:,4)))));
    s105 = (1 / (1+exp(-t(:,5)))) *(1-(1 / (1+exp(-t(:,5)))));
    s106 = (1 / (1+exp(-t(:,6)))) *(1-(1 / (1+exp(-t(:,6)))));
    s107 = (1 / (1+exp(-t(:,7)))) *(1-(1 / (1+exp(-t(:,7)))));
    s108 = (1 / (1+exp(-t(:,8)))) *(1-(1 / (1+exp(-t(:,8)))));
    s109 = (1 / (1+exp(-t(:,9)))) *(1-(1 / (1+exp(-t(:,9)))));
    s110 = (1 / (1+exp(-t(:,10))))*(1-(1 / (1+exp(-t(:,10)))));
    s111 = (1 / (1+exp(-t(:,11))))*(1-(1 / (1+exp(-t(:,11)))));
    s112 = (1 / (1+exp(-t(:,12))))*(1-(1 / (1+exp(-t(:,12)))));
    s113 = (1 / (1+exp(-t(:,13))))*(1-(1 / (1+exp(-t(:,13)))));
    s114 = (1 / (1+exp(-t(:,14))))*(1-(1 / (1+exp(-t(:,14)))));
    s115 = (1 / (1+exp(-t(:,15))))*(1-(1 / (1+exp(-t(:,15)))));
    s116 = (1 / (1+exp(-t(:,16))))*(1-(1 / (1+exp(-t(:,16)))));
       
    D1 = [s101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
          0,s102,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
          0,0,s103,0,0,0,0,0,0,0,0,0,0,0,0,0;
          0,0,0,s104,0,0,0,0,0,0,0,0,0,0,0,0;
          0,0,0,0,s105,0,0,0,0,0,0,0,0,0,0,0;
          0,0,0,0,0,s106,0,0,0,0,0,0,0,0,0,0;
          0,0,0,0,0,0,s107,0,0,0,0,0,0,0,0,0;
          0,0,0,0,0,0,0,s108,0,0,0,0,0,0,0,0;
          0,0,0,0,0,0,0,0,s109,0,0,0,0,0,0,0;
          0,0,0,0,0,0,0,0,0,s110,0,0,0,0,0,0;
          0,0,0,0,0,0,0,0,0,0,s111,0,0,0,0,0;
          0,0,0,0,0,0,0,0,0,0,0,s112,0,0,0,0;
          0,0,0,0,0,0,0,0,0,0,0,0,s113,0,0,0;
          0,0,0,0,0,0,0,0,0,0,0,0,0,s114,0,0;
          0,0,0,0,0,0,0,0,0,0,0,0,0,0,s115,0;
          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,s116];

    % backward pass - layer 2
    t = [out_layer1, 1] * W1;
    
    s201 = (1 / (1+exp(-t(1)))) *(1-(1 / (1+exp(-t(1)))));
    s202 = (1 / (1+exp(-t(2)))) *(1-(1 / (1+exp(-t(2)))));
    s203 = (1 / (1+exp(-t(3)))) *(1-(1 / (1+exp(-t(3)))));
    s204 = (1 / (1+exp(-t(4)))) *(1-(1 / (1+exp(-t(4)))));
    s205 = (1 / (1+exp(-t(5)))) *(1-(1 / (1+exp(-t(5)))));
    s206 = (1 / (1+exp(-t(6)))) *(1-(1 / (1+exp(-t(6)))));
    s207 = (1 / (1+exp(-t(7)))) *(1-(1 / (1+exp(-t(7)))));
    s208 = (1 / (1+exp(-t(8)))) *(1-(1 / (1+exp(-t(8)))));
    s209 = (1 / (1+exp(-t(9)))) *(1-(1 / (1+exp(-t(9)))));
    s210 = (1 / (1+exp(-t(10))))*(1-(1 / (1+exp(-t(10)))));
    
    D2                 = [s201,0,0,0,0,0,0,0,0,0;
                          0,s202,0,0,0,0,0,0,0,0;
                          0,0,s203,0,0,0,0,0,0,0;
                          0,0,0,s204,0,0,0,0,0,0;
                          0,0,0,0,s205,0,0,0,0,0;
                          0,0,0,0,0,s206,0,0,0,0;
                          0,0,0,0,0,0,s207,0,0,0;
                          0,0,0,0,0,0,0,s208,0,0;
                          0,0,0,0,0,0,0,0,s209,0;
                          0,0,0,0,0,0,0,0,0,s210];
    
    W2_                = W1(1:16,:);
    dW1                = -alpha*D1*W2_*D2*error'*D;
    dW2                = -alpha*D2*error'*[out_layer1, 1];
    W0                 = W0 + dW1';
    W1                 = W1 + dW2';

end
